"""Training loop for Slot Attention Transducer."""

import argparse
import json
import time
from pathlib import Path
from functools import partial

import yaml
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from data.vocab import CharVocab
from data.dataset import (
    load_english_merged, split_data, apply_training_regime,
    PastTenseDataset, collate_fn,
)
from model import SlotAttentionTransducer


def build_vocab_and_datasets(data_path: str, use_phonological: bool = True,
                             seed: int = 42, training_regime: str = "natural",
                             use_edits: bool = False, use_edit_labels: bool = False,
                             use_transducer: bool = False,
                             use_retrieval: bool = False, retrieval_k: int = 5,
                             use_class_retrieval: bool = False,
                             retrieval_mode: str = "knn"):
    """Load data, build vocab, create train/val/test datasets.

    Returns (vocab, train_ds, val_ds, test_ds, raw_train_entries) where
    raw_train_entries is the unregimed training split (useful for two-phase
    training that rebuilds the train_ds with a different regime mid-run).
    """
    entries = load_english_merged(data_path)
    train_entries, val_entries, test_entries = split_data(entries, seed=seed)
    raw_train_entries = list(train_entries)  # unregimed copy for phase-2 rebuild

    # Apply training regime (reorder/resample training data)
    train_entries = apply_training_regime(train_entries, training_regime, seed=seed)

    vocab = CharVocab()
    if use_phonological:
        all_seqs = [e["phon_src"] for e in entries] + [e["phon_tgt"] for e in entries]
    else:
        all_seqs = [list(e["orth_src"]) for e in entries] + [list(e["orth_tgt"]) for e in entries]
    vocab.build_from_sequences(all_seqs)

    # Build the kNN retrieval index over the (deduplicated) train pairs once.
    # Used by all three splits (train items get self-excluded by src match).
    retrieval_index = None
    if use_retrieval:
        from data.retrieval import RetrievalIndex
        seen_srcs = set()
        unique_pairs = []
        for e in raw_train_entries:
            src = e["phon_src"] if use_phonological else list(e["orth_src"])
            tgt = e["phon_tgt"] if use_phonological else list(e["orth_tgt"])
            key = tuple(src)
            if key in seen_srcs:
                continue
            seen_srcs.add(key)
            unique_pairs.append((src, tgt))
        retrieval_index = RetrievalIndex(unique_pairs, k=retrieval_k)

    train_ds = PastTenseDataset(train_entries, vocab, use_phonological,
                                use_edits=use_edits, use_edit_labels=use_edit_labels,
                                use_transducer=use_transducer,
                                retrieval_index=retrieval_index,
                                use_class_retrieval=use_class_retrieval,
                                retrieval_mode=retrieval_mode)
    val_ds = PastTenseDataset(val_entries, vocab, use_phonological,
                              use_edits=use_edits, use_edit_labels=use_edit_labels,
                              use_transducer=use_transducer,
                              retrieval_index=retrieval_index,
                              use_class_retrieval=use_class_retrieval,
                              retrieval_mode=retrieval_mode)
    test_ds = PastTenseDataset(test_entries, vocab, use_phonological,
                               use_edits=use_edits, use_edit_labels=use_edit_labels,
                               use_transducer=use_transducer,
                               retrieval_index=retrieval_index,
                               use_class_retrieval=use_class_retrieval,
                              retrieval_mode=retrieval_mode)
    return vocab, train_ds, val_ds, test_ds, raw_train_entries


def plot_losses(history: dict, results_dir: Path):
    """Save loss curves to results_dir."""
    epochs = history["epoch"]

    # Combined loss plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Total Loss")
    axes[0].set_title("Train / Val Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_transduce"], label="train")
    axes[1].plot(epochs, history["val_transduce"], label="val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Transduction Loss")
    axes[1].set_title("Transduction Loss (NLL)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    if any(v != 0 for v in history["l0"]):
        axes[2].plot(epochs, history["l0"], color="tab:orange")
        axes[2].set_ylabel("L0 (expected active slots)")
        axes[2].set_title("L0 Regularization")
    else:
        axes[2].text(0.5, 0.5, "No slots\n(baseline)", ha="center", va="center",
                     transform=axes[2].transAxes, fontsize=12, color="gray")
        axes[2].set_title("L0 Regularization (N/A)")
    axes[2].set_xlabel("Epoch")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(results_dir / "loss_curves.png", dpi=150)
    plt.close(fig)

    # If multi-task, also plot recon loss
    if history.get("recon_loss"):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(epochs, history["recon_loss"])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Reconstruction Loss")
        ax.set_title("Reconstruction Loss (Multi-task)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(results_dir / "recon_loss.png", dpi=150)
        plt.close(fig)


def _unpack_retrieval(batch, device):
    """Pull (retrieval_ids, retrieval_pad_mask) off a batch tuple if present.

    Batch shapes (only retrieval-relevant ones):
      - len 5, batch[3].dim()==3: non-transducer + retrieval
            (src, tgt, lbl, retr_ids, retr_mask)
      - len 6, batch[3].dim()==3: non-transducer + retrieval + cluster_id
            (src, tgt, lbl, retr_ids, retr_mask, cluster_id)
      - len 6, batch[3].dim()==2: transducer + retrieval
            (src, tgt, lbl, action, retr_ids, retr_mask)
    """
    if len(batch) == 6 and batch[3].dim() == 3:
        return batch[3].to(device), batch[4].to(device)
    if len(batch) == 6:
        return batch[4].to(device), batch[5].to(device)
    if len(batch) == 5 and batch[3].dim() == 3:
        return batch[3].to(device), batch[4].to(device)
    return None, None


def compute_accuracy(model, dataloader, vocab, device):
    """Compute exact-match accuracy via greedy decoding."""
    model.eval()
    correct = 0
    total = 0
    for batch in dataloader:
        src, tgt = batch[0].to(device), batch[1].to(device)
        retr_ids, retr_mask = _unpack_retrieval(batch, device)
        preds = model.greedy_decode(src, sos_idx=vocab.sos_idx, eos_idx=vocab.eos_idx,
                                     retrieval_ids=retr_ids,
                                     retrieval_pad_mask=retr_mask)
        for i in range(src.size(0)):
            pred_tokens = vocab.decode(preds[i].tolist())
            tgt_tokens = vocab.decode(tgt[i].tolist())
            if pred_tokens == tgt_tokens:
                correct += 1
            total += 1
    return correct / total if total > 0 else 0.0


@torch.no_grad()
def _retrieval_for_predicted_cluster(model, src_chars, vocab, device, retrieval_index):
    """Predict cluster from src, retrieve top-k within that cluster, build
    retrieval memory tensors. Used by predicted-cluster eval (v30b)."""
    from data.dataset import _pad_retrieved
    src_ids = vocab.encode(src_chars, add_eos=True)
    src_t = torch.tensor([src_ids], dtype=torch.long, device=device)
    cid = model.predict_clusters(src_t).item()
    nbrs = retrieval_index.query_in_cluster(src_chars, cid)
    nbr_tgts = retrieval_index.get_targets(nbrs)
    retr_id_tensors = [
        torch.tensor(vocab.encode(t, add_sos=False, add_eos=False),
                     dtype=torch.long)
        for t in nbr_tgts
    ]
    while len(retr_id_tensors) < retrieval_index.k:
        retr_id_tensors.append(retr_id_tensors[-1] if retr_id_tensors
                               else torch.tensor([vocab.pad_idx], dtype=torch.long))
    retr_ids, retr_mask = _pad_retrieved([retr_id_tensors], vocab.pad_idx)
    return src_t, retr_ids.to(device), retr_mask.to(device), cid


@torch.no_grad()
def compute_accuracy_predicted_cluster(model, entries, vocab, device, retrieval_index,
                                        use_phon: bool = True):
    """Per-sample eval: predict cluster from src, retrieve in predicted cluster, decode.
    The unbiased eval path for v30b (no tgt info used at retrieval time)."""
    model.eval()
    correct = reg_correct = irr_correct = 0
    reg_total = irr_total = 0
    for e in entries:
        src_chars = e["phon_src"] if use_phon else list(e["orth_src"])
        tgt_chars = e["phon_tgt"] if use_phon else list(e["orth_tgt"])
        src_t, retr_ids, retr_mask, _cid = _retrieval_for_predicted_cluster(
            model, src_chars, vocab, device, retrieval_index)
        preds = model.greedy_decode(src_t, sos_idx=vocab.sos_idx, eos_idx=vocab.eos_idx,
                                    retrieval_ids=retr_ids,
                                    retrieval_pad_mask=retr_mask)
        pred = vocab.decode(preds[0].tolist())
        ok = pred == tgt_chars
        correct += int(ok)
        if e.get("regularity") == "irreg":
            irr_correct += int(ok); irr_total += 1
        else:
            reg_correct += int(ok); reg_total += 1
    n = len(entries)
    return (
        reg_correct / reg_total if reg_total else 0.0,
        irr_correct / irr_total if irr_total else 0.0,
        correct / n if n else 0.0,
    )


@torch.no_grad()
def compute_accuracy_split(model, dataloader, vocab, device):
    """Per-class exact-match accuracy: (reg, irr, all).

    Used to track the U-shape during two-phase (Rumelhart & McClelland) training:
    regular accuracy should rise smoothly, irregular accuracy should peak at end of
    phase 1, dip when phase 2 starts (overregularization), then partially recover.
    """
    model.eval()
    reg_correct = reg_total = irr_correct = irr_total = 0
    for batch in dataloader:
        src, tgt = batch[0].to(device), batch[1].to(device)
        reg_labels = batch[2]
        retr_ids, retr_mask = _unpack_retrieval(batch, device)
        preds = model.greedy_decode(src, sos_idx=vocab.sos_idx, eos_idx=vocab.eos_idx,
                                     retrieval_ids=retr_ids,
                                     retrieval_pad_mask=retr_mask)
        for i in range(src.size(0)):
            ok = vocab.decode(preds[i].tolist()) == vocab.decode(tgt[i].tolist())
            if reg_labels[i].item() == 0:
                reg_correct += int(ok)
                reg_total += 1
            else:
                irr_correct += int(ok)
                irr_total += 1
    reg_acc = reg_correct / reg_total if reg_total else 0.0
    irr_acc = irr_correct / irr_total if irr_total else 0.0
    all_acc = (reg_correct + irr_correct) / max(1, reg_total + irr_total)
    return reg_acc, irr_acc, all_acc


def log(msg: str):
    """Print with immediate flush for piped output."""
    print(msg, flush=True)


def _strip_specials(src: torch.Tensor, tgt: torch.Tensor,
                    pad_idx: int, sos_idx: int, eos_idx: int):
    """Per-sample raw int lists with PAD/SOS/EOS removed (for HMNT oracle)."""
    src_lists = []
    tgt_lists = []
    specials = {pad_idx, sos_idx, eos_idx}
    for s in src:
        src_lists.append([int(x) for x in s.tolist() if int(x) not in specials])
    for t in tgt:
        tgt_lists.append([int(x) for x in t.tolist() if int(x) not in specials])
    return src_lists, tgt_lists


def train(config: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")

    # Data
    regime = config.get("training_regime", "natural")
    phase1_epochs = config.get("phase1_epochs", 0)
    phase1_regime = config.get("phase1_regime", "balanced")
    use_edits = config.get("decoder_type") in ("edit", "population_edit")
    use_edit_labels = config.get("decoder_type") == "edit_labeler"
    use_transducer = config.get("decoder_type") == "transducer"
    use_retrieval = config.get("use_retrieval", False)
    retrieval_k = config.get("retrieval_k", 5)
    use_class_retrieval = config.get("use_class_retrieval", False)
    retrieval_mode = config.get("retrieval_mode", "knn")
    use_cluster_predictor = config.get("use_cluster_predictor", False)
    lambda_cluster = config.get("lambda_cluster", 0.5)
    seed = config.get("seed", 42)
    use_phon = config.get("use_phonological", True)
    vocab, train_ds, val_ds, test_ds, raw_train_entries = build_vocab_and_datasets(
        config["data_path"],
        use_phonological=use_phon,
        seed=seed,
        training_regime=regime,
        use_edits=use_edits,
        use_edit_labels=use_edit_labels,
        use_transducer=use_transducer,
        use_retrieval=use_retrieval,
        retrieval_k=retrieval_k,
        use_class_retrieval=use_class_retrieval,
        retrieval_mode=retrieval_mode,
    )

    pad_idx = vocab.pad_idx
    collate = partial(collate_fn, pad_idx=pad_idx)

    # Two-phase (Rumelhart & McClelland 1986) training: start with a small
    # high-irregular-density vocabulary, then expand to the full set.
    if phase1_epochs > 0:
        phase1_entries = apply_training_regime(
            list(raw_train_entries), phase1_regime, seed=seed)
        phase1_ds = PastTenseDataset(phase1_entries, vocab, use_phon,
                                     use_edits=use_edits, use_edit_labels=use_edit_labels,
                                     use_transducer=use_transducer)
        log(f"Phase 1 ({phase1_regime}): {len(phase1_ds)} examples for {phase1_epochs} epochs")
        log(f"Phase 2 ({regime}): {len(train_ds)} examples thereafter")
        train_loader = DataLoader(phase1_ds, batch_size=config["batch_size"],
                                  shuffle=True, collate_fn=collate)
        phase2_loader = DataLoader(train_ds, batch_size=config["batch_size"],
                                   shuffle=True, collate_fn=collate)
    else:
        log(f"Training regime: {regime}")
        train_loader = DataLoader(train_ds, batch_size=config["batch_size"],
                                  shuffle=True, collate_fn=collate)
        phase2_loader = None

    val_loader = DataLoader(val_ds, batch_size=config["batch_size"],
                            shuffle=False, collate_fn=collate)

    # Model
    l0_mode = config.get("l0_mode", "conditional")
    use_slots = config.get("use_slots", True)
    model = SlotAttentionTransducer(
        vocab_size=len(vocab),
        d_model=config.get("d_model", 128),
        nhead=config.get("nhead", 4),
        enc_layers=config.get("enc_layers", 3),
        dec_layers=config.get("dec_layers", 3),
        d_ff=config.get("d_ff", 256),
        num_slots=config.get("num_slots", 8),
        slot_iters=config.get("slot_iters", 3),
        dropout=config.get("dropout", 0.1),
        pad_idx=pad_idx,
        lambda_l0=config.get("lambda_l0", 0.01),
        alpha_recon=config.get("alpha_recon", 0.0),
        l0_beta=config.get("l0_beta", 0.66),
        l0_mode=l0_mode,
        target_l0=config.get("target_l0", 2.0),
        lagrangian_lr=config.get("lagrangian_lr", 0.01),
        use_copy=config.get("use_copy", False),
        slot_nhead=config.get("slot_nhead", 1),
        use_slots=use_slots,
        decoder_type=config.get("decoder_type", "transformer"),
        encoder_type=config.get("encoder_type", "transformer"),
        alpha_cls=config.get("alpha_cls", 0.0),
        dec_bottleneck=config.get("dec_bottleneck", 0),
        lstm_hidden=config.get("lstm_hidden", 0),
        num_experts=config.get("num_experts", 4),
        expert_hidden=config.get("expert_hidden", 64),
        routing_mode=config.get("routing_mode", "soft"),
        gumbel_tau=config.get("gumbel_tau", 1.0),
        lambda_balance=config.get("lambda_balance", 0.01),
        alpha_cls_router=config.get("alpha_cls_router", 0.0),
        lambda_diversity=config.get("lambda_diversity", 0.0),
        confidence_mode=config.get("confidence_mode", "input"),
        neuron_dropout=config.get("neuron_dropout", 0.0),
        confidence_tau=config.get("confidence_tau", 1.0),
        mono_alpha_init=config.get("mono_alpha_init", 1.0),
        use_retrieval=use_retrieval,
        num_clusters=(train_ds.retrieval_index.num_clusters
                      if use_cluster_predictor and train_ds.retrieval_index is not None
                      else 0),
        lambda_cluster=lambda_cluster,
    ).to(device)
    # Stash num_clusters in config so the eval pass (separate process) can
    # rebuild the model with the same architecture.
    if use_cluster_predictor and train_ds.retrieval_index is not None:
        config["_num_clusters"] = train_ds.retrieval_index.num_clusters

    log(f"Vocab size: {len(vocab)}")
    log(f"Use slots: {use_slots}")
    if use_slots:
        log(f"L0 mode: {l0_mode}")
    log(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    log(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # --- Phase 0: Pretrain encoder+slots with reconstruction (optional) ---
    pretrain_epochs = config.get("pretrain_epochs", 0)
    if pretrain_epochs > 0 and use_slots:
        log(f"\n=== Pretrain Phase: {pretrain_epochs} epochs (encoder+slots+recon) ===")
        # Build a temporary reconstruction decoder if not already present
        if model.recon_decoder is None:
            from model.decoder import TransformerCharDecoder
            model.recon_decoder = TransformerCharDecoder(
                vocab_size=len(vocab),
                d_model=config.get("d_model", 128),
                nhead=config.get("nhead", 4),
                num_layers=config.get("dec_layers", 3),
                d_ff=config.get("d_ff", 256),
                dropout=config.get("dropout", 0.1),
                pad_idx=pad_idx,
            ).to(device)
            _temp_recon = True
        else:
            _temp_recon = False

        # Only train encoder + slot_attention + l0drop + recon_decoder
        pretrain_params = (
            list(model.encoder.parameters())
            + list(model.slot_attention.parameters())
            + list(model.l0drop.parameters())
            + list(model.recon_decoder.parameters())
        )
        pretrain_opt = AdamW(pretrain_params, lr=config.get("lr", 3e-4),
                             weight_decay=config.get("weight_decay", 1e-4))

        for ep in range(1, pretrain_epochs + 1):
            model.train()
            ep_loss = 0.0
            for src, *_ in train_loader:
                src = src.to(device)
                H = model.encoder(src)
                slots = model.slot_attention(H)
                memory = model.l0drop(slots)
                # Autoencoding: reconstruct source from slots
                sos = torch.full((src.size(0), 1), vocab.sos_idx,
                                 dtype=torch.long, device=device)
                eos = torch.full((src.size(0), 1), vocab.eos_idx,
                                 dtype=torch.long, device=device)
                recon_tgt = torch.cat([sos, src, eos], dim=1)
                recon_input = recon_tgt[:, :-1]
                recon_target = recon_tgt[:, 1:]
                recon_logits = model.recon_decoder(recon_input, memory)
                loss_recon = F.cross_entropy(
                    recon_logits.reshape(-1, recon_logits.size(-1)),
                    recon_target.reshape(-1),
                    ignore_index=pad_idx,
                )
                loss_l0 = model.l0drop.l0_loss()
                loss = loss_recon + loss_l0
                pretrain_opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(pretrain_params, 1.0)
                pretrain_opt.step()
                ep_loss += loss.item()

            if use_slots and l0_mode == "conditional" and hasattr(model.l0drop, "update_lagrangian"):
                model.l0drop.update_lagrangian()

            avg = ep_loss / len(train_loader)
            l0_val = loss_l0.item()
            log(f"  Pretrain {ep:3d}/{pretrain_epochs} | recon_loss {avg:.4f} | L0 {l0_val:.4f}")

        # Remove temp recon decoder to free memory (slots are now pretrained)
        if _temp_recon:
            del model.recon_decoder
            model.recon_decoder = None

        log("=== Pretrain complete, starting transduction training ===\n")

    optimizer = AdamW(model.parameters(), lr=config.get("lr", 3e-4),
                      weight_decay=config.get("weight_decay", 1e-4))
    scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"])

    save_dir = Path(config.get("save_dir", "checkpoints"))
    save_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(config.get("results_dir", "results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    plot_every = config.get("plot_every", 5)
    patience = config.get("patience", 0)  # 0 = no early stopping
    best_val_loss = float("inf")
    best_val_acc = -1.0  # used as the checkpoint criterion when track_val_split is on
    epochs_without_improvement = 0
    # When greedy-decode val accuracy is tracked, use it (not val_loss) to pick
    # the best checkpoint. With teacher-forced decoders the model often hits
    # min val_loss long before peak greedy-decode accuracy, so the val_loss
    # criterion saves the wrong epoch — confirmed empirically on v22a/v23a/b.
    use_val_acc_criterion = config.get("track_val_split", False)

    # L0 warmup schedule
    lambda_l0_target = config.get("lambda_l0", 0.01)
    l0_warmup_epochs = config.get("l0_warmup_epochs", 0)

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_transduce": [],
        "val_transduce": [],
        "l0": [],
        "lambda_l0_effective": [],
        "recon_loss": [],
        "balance_loss": [],
        "val_reg_acc": [],
        "val_irr_acc": [],
        "val_acc": [],
        "phase": [],
    }

    # Gumbel temperature annealing
    decoder_type = config.get("decoder_type", "transformer")
    gumbel_tau_start = config.get("gumbel_tau", 1.0)
    gumbel_tau_end = config.get("gumbel_tau_end", 0.1)
    gumbel_anneal_epochs = config.get("gumbel_anneal_epochs", 0)  # 0 = no annealing

    # Router cls auxiliary annealing: anneal alpha_cls_router to 0 over N epochs
    alpha_cls_router_init = config.get("alpha_cls_router", 0.0)
    cls_anneal_epochs = config.get("cls_anneal_epochs", 0)  # 0 = no annealing (constant)

    # Confidence temperature annealing for population decoder
    confidence_tau_start = config.get("confidence_tau", 1.0)
    confidence_tau_end = config.get("confidence_tau_end", 0.0)  # 0 = no annealing
    confidence_tau_anneal_epochs = config.get("confidence_tau_anneal_epochs", 0)

    # DAgger schedule for HMNT transducer
    dagger_start_epoch = config.get("dagger_start_epoch", 0)        # 0 = pure TF
    dagger_beta_max = config.get("dagger_beta_max", 0.0)            # 0 = no DAgger
    dagger_anneal_epochs = config.get("dagger_anneal_epochs", 0)    # epochs to anneal 0->max

    # L0 annealing schedule: Phase 1 (all slots open) → Phase 2 (anneal to target)
    l0_anneal_start = config.get("l0_anneal_start", 0)  # epoch to begin annealing
    l0_anneal_epochs = config.get("l0_anneal_epochs", 0)  # epochs over which to anneal
    final_target_l0 = config.get("target_l0", 2.0)
    num_slots = config.get("num_slots", 8)

    for epoch in range(1, config["epochs"] + 1):
        # Two-phase regimen: swap to full vocabulary at the boundary
        if phase2_loader is not None and epoch == phase1_epochs + 1:
            log(f"  ← phase 1 → phase 2 transition at epoch {epoch}")
            train_loader = phase2_loader
            # Reset best-val tracking so early stopping doesn't fire from phase-1
            # plateau, and so we save a phase-2 best checkpoint.
            best_val_loss = float("inf")
            epochs_without_improvement = 0

        # Compute effective lambda_l0 with warmup
        if l0_warmup_epochs > 0 and epoch <= l0_warmup_epochs:
            effective_lambda_l0 = lambda_l0_target * (epoch / l0_warmup_epochs)
        else:
            effective_lambda_l0 = lambda_l0_target
        model.lambda_l0 = effective_lambda_l0

        # L0 target annealing (developmental schedule)
        if l0_anneal_start > 0 and l0_anneal_epochs > 0 and use_slots and l0_mode == "conditional":
            if epoch < l0_anneal_start:
                # Phase 1: all slots open
                effective_target = float(num_slots)
            elif epoch < l0_anneal_start + l0_anneal_epochs:
                # Phase 2: linear anneal from num_slots → final target
                progress = (epoch - l0_anneal_start) / l0_anneal_epochs
                effective_target = num_slots + progress * (final_target_l0 - num_slots)
            else:
                # Phase 3: hold at final target
                effective_target = final_target_l0
            model.l0drop.target_l0 = effective_target
            # Reset best val loss when annealing begins so early stopping
            # doesn't trigger immediately (Phase 2 loss will be worse than Phase 1)
            if epoch == l0_anneal_start:
                best_val_loss = float("inf")
                epochs_without_improvement = 0

        # When DAgger engages, the training distribution shifts (model sees
        # divergent states from its own rollouts), so val_loss typically spikes
        # before recovering. Reset best-val tracking so we save a fresh
        # post-DAgger checkpoint and don't early-stop on the spike.
        if (use_transducer and dagger_beta_max > 0
                and epoch == dagger_start_epoch):
            best_val_loss = float("inf")
            epochs_without_improvement = 0

        # Gumbel temperature annealing for MoE
        if decoder_type == "moe_lstm" and gumbel_anneal_epochs > 0:
            progress = min(epoch / gumbel_anneal_epochs, 1.0)
            tau = gumbel_tau_start + progress * (gumbel_tau_end - gumbel_tau_start)
            model.decoder.router.tau = tau

        # Router cls auxiliary annealing: linear decay to 0
        if decoder_type == "moe_lstm" and cls_anneal_epochs > 0 and alpha_cls_router_init > 0:
            progress = min(epoch / cls_anneal_epochs, 1.0)
            model.decoder.alpha_cls_router = alpha_cls_router_init * (1.0 - progress)

        # Scheduled sampling annealing for edit decoder
        ss_target = config.get("scheduled_sampling", 0.0)
        ss_anneal = config.get("ss_anneal_epochs", 0)
        if decoder_type == "edit" and ss_anneal > 0 and hasattr(model.decoder, 'scheduled_sampling'):
            progress = min(epoch / ss_anneal, 1.0)
            model.decoder.scheduled_sampling = ss_target * progress

        # Confidence temperature annealing for population decoder
        if decoder_type == "population" and confidence_tau_anneal_epochs > 0:
            progress = min(epoch / confidence_tau_anneal_epochs, 1.0)
            tau = confidence_tau_start + progress * (confidence_tau_end - confidence_tau_start)
            model.decoder.confidence_tau = tau

        model.train()
        epoch_loss = 0.0
        epoch_transduce = 0.0
        epoch_recon = 0.0
        epoch_balance = 0.0
        t0 = time.time()

        # DAgger β for this epoch (HMNT only)
        if use_transducer and dagger_beta_max > 0 and epoch >= dagger_start_epoch:
            if dagger_anneal_epochs > 0:
                progress = min(
                    (epoch - dagger_start_epoch) / dagger_anneal_epochs, 1.0)
            else:
                progress = 1.0
            dagger_beta = max(0.0, dagger_beta_max * progress)
        else:
            dagger_beta = 0.0

        for batch in train_loader:
            edit_tgt = None
            edit_lbl = None
            suffix_tgt = None
            action_tgt = None
            retr_ids = None
            retr_mask = None
            if use_edit_labels:
                src, tgt, reg_labels, edit_lbl, suffix_tgt = batch
                edit_lbl = edit_lbl.to(device)
                suffix_tgt = suffix_tgt.to(device)
            elif use_edits:
                src, tgt, reg_labels, edit_tgt = batch
                edit_tgt = edit_tgt.to(device)
            elif use_transducer and use_retrieval:
                src, tgt, reg_labels, action_tgt, retr_ids, retr_mask = batch
                action_tgt = action_tgt.to(device)
                retr_ids = retr_ids.to(device)
                retr_mask = retr_mask.to(device)
            elif use_transducer:
                src, tgt, reg_labels, action_tgt = batch
                action_tgt = action_tgt.to(device)
            elif use_retrieval and use_cluster_predictor:
                # 6-field: (src, tgt, label, retr_ids, retr_mask, cluster_id)
                src, tgt, reg_labels, retr_ids, retr_mask, cluster_targets = batch
                retr_ids = retr_ids.to(device)
                retr_mask = retr_mask.to(device)
                cluster_targets = cluster_targets.to(device)
            elif use_retrieval:
                # Dataset always appends cluster_id when retrieval_index has
                # clusters (always now). Tolerate both 5-field (legacy) and
                # 6-field shapes; ignore cluster_id when the predictor is off.
                if len(batch) == 6:
                    src, tgt, reg_labels, retr_ids, retr_mask, _ = batch
                else:
                    src, tgt, reg_labels, retr_ids, retr_mask = batch
                retr_ids = retr_ids.to(device)
                retr_mask = retr_mask.to(device)
                cluster_targets = None
            else:
                src, tgt, reg_labels = batch
                cluster_targets = None
            src, tgt = src.to(device), tgt.to(device)
            reg_labels = reg_labels.to(device)
            # Build reconstruction input if multi-task is enabled
            src_for_recon = None
            if model.recon_decoder is not None:
                sos = torch.full((src.size(0), 1), vocab.sos_idx,
                                 dtype=torch.long, device=device)
                eos = torch.full((src.size(0), 1), vocab.eos_idx,
                                 dtype=torch.long, device=device)
                src_for_recon = torch.cat([sos, src, eos], dim=1)
            # For DAgger oracle re-querying: per-sample raw lists.
            src_no_eos = None
            tgt_no_special = None
            if use_transducer and dagger_beta > 0:
                src_no_eos, tgt_no_special = _strip_specials(
                    src, tgt, vocab.pad_idx, vocab.sos_idx, vocab.eos_idx)
            result = model(src, tgt, src_for_recon=src_for_recon,
                           reg_labels=reg_labels, edit_targets=edit_tgt,
                           edit_labels=edit_lbl, suffix_targets=suffix_tgt,
                           action_targets=action_tgt,
                           dagger_beta=dagger_beta,
                           src_no_eos=src_no_eos,
                           tgt_no_special=tgt_no_special,
                           retrieval_ids=retr_ids,
                           retrieval_pad_mask=retr_mask,
                           cluster_targets=cluster_targets)
            loss = result["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            epoch_transduce += result["loss_transduce"].item()
            if "loss_recon" in result:
                epoch_recon += result["loss_recon"].item()
            if "loss_balance" in result:
                epoch_balance += result["loss_balance"].item()

        scheduler.step()
        n_batches = len(train_loader)
        avg_train_loss = epoch_loss / n_batches
        avg_train_transduce = epoch_transduce / n_batches

        # Validation
        model.eval()
        val_loss = 0.0
        val_transduce = 0.0
        with torch.no_grad():
            for batch in val_loader:
                edit_tgt = None
                edit_lbl = None
                suffix_tgt = None
                action_tgt = None
                retr_ids = None
                retr_mask = None
                if use_edit_labels:
                    src, tgt, reg_labels, edit_lbl, suffix_tgt = batch
                    edit_lbl = edit_lbl.to(device)
                    suffix_tgt = suffix_tgt.to(device)
                elif use_edits:
                    src, tgt, reg_labels, edit_tgt = batch
                    edit_tgt = edit_tgt.to(device)
                elif use_transducer and use_retrieval:
                    src, tgt, reg_labels, action_tgt, retr_ids, retr_mask = batch
                    action_tgt = action_tgt.to(device)
                    retr_ids = retr_ids.to(device)
                    retr_mask = retr_mask.to(device)
                elif use_transducer:
                    src, tgt, reg_labels, action_tgt = batch
                    action_tgt = action_tgt.to(device)
                elif use_retrieval and use_cluster_predictor:
                    src, tgt, reg_labels, retr_ids, retr_mask, _cluster = batch
                    retr_ids = retr_ids.to(device)
                    retr_mask = retr_mask.to(device)
                elif use_retrieval:
                    if len(batch) == 6:
                        src, tgt, reg_labels, retr_ids, retr_mask, _ = batch
                    else:
                        src, tgt, reg_labels, retr_ids, retr_mask = batch
                    retr_ids = retr_ids.to(device)
                    retr_mask = retr_mask.to(device)
                else:
                    src, tgt, reg_labels = batch
                src, tgt = src.to(device), tgt.to(device)
                reg_labels = reg_labels.to(device)
                result = model(src, tgt, reg_labels=reg_labels, edit_targets=edit_tgt,
                               edit_labels=edit_lbl, suffix_targets=suffix_tgt,
                               action_targets=action_tgt,
                               retrieval_ids=retr_ids,
                               retrieval_pad_mask=retr_mask)
                val_loss += result["loss"].item()
                if "loss_transduce" in result:
                    val_transduce += result["loss_transduce"].item()
        avg_val_loss = val_loss / len(val_loader)
        avg_val_transduce = val_transduce / len(val_loader)

        # Update Lagrangian multiplier (input-conditional mode only)
        if use_slots and l0_mode == "conditional" and hasattr(model.l0drop, "update_lagrangian"):
            model.l0drop.update_lagrangian()

        elapsed = time.time() - t0
        l0_val = result["loss_l0"].item()
        if decoder_type in ("edit", "population_edit", "edit_labeler"):
            div_val = result.get("loss_diversity", torch.tensor(0.0)).item()
            div_str = f" | div {div_val:.4f}" if div_val > 0 else ""
            log(f"Epoch {epoch:3d} | train_loss {avg_train_loss:.4f} | "
                f"val_loss {avg_val_loss:.4f}{div_str} | {elapsed:.1f}s")
        elif decoder_type == "population":
            div_val = result.get("loss_diversity", torch.tensor(0.0)).item()
            bal_val = result.get("loss_balance", torch.tensor(0.0)).item()
            tau_str = f" | tau {model.decoder.confidence_tau:.3f}" if confidence_tau_anneal_epochs > 0 else ""
            log(f"Epoch {epoch:3d} | train_loss {avg_train_loss:.4f} | "
                f"val_loss {avg_val_loss:.4f} | div {div_val:.4f} | bal {bal_val:.4f}{tau_str} | {elapsed:.1f}s")
        elif decoder_type == "moe_lstm":
            avg_balance = epoch_balance / n_batches
            tau_str = f" | tau {model.decoder.router.tau:.3f}" if config.get("routing_mode") == "gumbel" else ""
            div_str = f" | div {result['loss_diversity'].item():.4f}" if result.get("loss_diversity", torch.tensor(0.0)).item() > 0 else ""
            log(f"Epoch {epoch:3d} | train_loss {avg_train_loss:.4f} | "
                f"val_loss {avg_val_loss:.4f} | balance {avg_balance:.4f}{tau_str}{div_str} | {elapsed:.1f}s")
        elif use_transducer:
            beta_str = f" | β {dagger_beta:.3f}" if dagger_beta > 0 else ""
            log(f"Epoch {epoch:3d} | train_loss {avg_train_loss:.4f} | "
                f"val_loss {avg_val_loss:.4f}{beta_str} | {elapsed:.1f}s")
        elif not use_slots:
            log(f"Epoch {epoch:3d} | train_loss {avg_train_loss:.4f} | "
                f"val_loss {avg_val_loss:.4f} | {elapsed:.1f}s")
        elif l0_mode == "conditional":
            lagr_lambda = model.l0drop.lagrangian_lambda.item()
            cur_target = model.l0drop.target_l0
            if isinstance(cur_target, float):
                target_str = f"{cur_target:.1f}"
            else:
                target_str = f"{cur_target:.1f}"
            log(f"Epoch {epoch:3d} | train_loss {avg_train_loss:.4f} | "
                f"val_loss {avg_val_loss:.4f} | L0_loss {l0_val:.4f} | "
                f"target_l0 {target_str} | λ_lagr {lagr_lambda:.4f} | {elapsed:.1f}s")
        else:
            log(f"Epoch {epoch:3d} | train_loss {avg_train_loss:.4f} | "
                f"val_loss {avg_val_loss:.4f} | L0 {l0_val:.2f} | "
                f"λ_l0 {effective_lambda_l0:.4f} | {elapsed:.1f}s")

        # Per-class val accuracy (for U-shape plotting in two-phase training).
        # Only run if phase1 is enabled OR explicitly requested, since this
        # adds ~5s/epoch from greedy decoding on val.
        track_split = phase2_loader is not None or config.get("track_val_split", False)
        if track_split:
            if use_cluster_predictor and train_ds.retrieval_index is not None:
                # v30b: predicted-cluster eval (closes the val/test info leak).
                val_reg, val_irr, val_all = compute_accuracy_predicted_cluster(
                    model, val_ds.entries, vocab, device, train_ds.retrieval_index,
                    use_phon=use_phon)
            else:
                val_reg, val_irr, val_all = compute_accuracy_split(
                    model, val_loader, vocab, device)
            history["val_reg_acc"].append(val_reg)
            history["val_irr_acc"].append(val_irr)
            history["val_acc"].append(val_all)
            log(f"           val_acc all={val_all:.3f} reg={val_reg:.3f} irr={val_irr:.3f}")

        cur_phase = 1 if (phase2_loader is not None and epoch <= phase1_epochs) else (
            2 if phase2_loader is not None else 0)
        history["phase"].append(cur_phase)

        # Track history
        history["epoch"].append(epoch)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_transduce"].append(avg_train_transduce)
        history["val_transduce"].append(avg_val_transduce)
        history["l0"].append(l0_val)
        history["lambda_l0_effective"].append(effective_lambda_l0)
        if "loss_recon" in result:
            history["recon_loss"].append(epoch_recon / n_batches)
        if decoder_type == "moe_lstm":
            history["balance_loss"].append(epoch_balance / n_batches)

        # Save plots periodically and on last epoch
        if epoch % plot_every == 0 or epoch == config["epochs"]:
            plot_losses(history, results_dir)
            with open(results_dir / "history.json", "w") as f:
                json.dump(history, f, indent=2)

        # Best-checkpoint selection: greedy-decode val_acc (when tracked)
        # is the right signal; val_loss alone tends to bottom out before
        # decoding accuracy peaks.
        if use_val_acc_criterion and history["val_acc"]:
            cur_val_acc = history["val_acc"][-1]
            improved = cur_val_acc > best_val_acc
            if improved:
                best_val_acc = cur_val_acc
        else:
            improved = avg_val_loss < best_val_loss
            if improved:
                best_val_loss = avg_val_loss

        if improved:
            epochs_without_improvement = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": avg_val_loss,
                "val_acc": (history["val_acc"][-1] if history["val_acc"] else None),
                "vocab": vocab,
                "config": config,
            }, save_dir / "best_model.pt")
        else:
            epochs_without_improvement += 1

        # Early stopping (disabled until after anneal begins, so model doesn't
        # stop before pruning has a chance to take effect; also disabled until
        # after phase 1 so a flat phase-1 val curve doesn't kill the run).
        # Disable early stopping until DAgger has fully ramped (its β-mixing
        # period is the actual training phase we care about). Without this the
        # best-val checkpoint freezes at the pre-DAgger plateau before the
        # exposure-bias fix has a chance to take effect.
        dagger_end = (dagger_start_epoch + dagger_anneal_epochs
                      if use_transducer and dagger_beta_max > 0 else 0)
        early_stop_after = max(
            l0_anneal_start if l0_anneal_start > 0 else 0,
            phase1_epochs if phase2_loader is not None else 0,
            dagger_end,
        )
        if patience > 0 and epochs_without_improvement >= patience and epoch > early_stop_after:
            log(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            # Save final plots
            plot_losses(history, results_dir)
            with open(results_dir / "history.json", "w") as f:
                json.dump(history, f, indent=2)
            break

    # Load best model for final evaluation
    best_ckpt = torch.load(save_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    val_acc = best_ckpt.get("val_acc")
    if val_acc is not None:
        log(f"Loaded best model from epoch {best_ckpt['epoch']} "
            f"(val_acc={val_acc:.4f}, val_loss={best_ckpt['val_loss']:.4f})")
    else:
        log(f"Loaded best model from epoch {best_ckpt['epoch']} "
            f"(val_loss={best_ckpt['val_loss']:.4f})")

    # Final test accuracy
    if use_cluster_predictor and train_ds.retrieval_index is not None:
        test_reg, test_irr, test_acc = compute_accuracy_predicted_cluster(
            model, test_ds.entries, vocab, device, train_ds.retrieval_index,
            use_phon=use_phon)
        log(f"\nTest exact-match accuracy: {test_acc:.4f} "
            f"(reg={test_reg:.4f} irr={test_irr:.4f}, predicted-cluster eval)")
    else:
        test_loader = DataLoader(test_ds, batch_size=config["batch_size"],
                                 shuffle=False, collate_fn=collate)
        test_acc = compute_accuracy(model, test_loader, vocab, device)
        log(f"\nTest exact-match accuracy: {test_acc:.4f}")
    log(f"Loss curves saved to {results_dir}/loss_curves.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train(config)
