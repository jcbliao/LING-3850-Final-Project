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
                             use_edits: bool = False, use_edit_labels: bool = False):
    """Load data, build vocab, create train/val/test datasets."""
    entries = load_english_merged(data_path)
    train_entries, val_entries, test_entries = split_data(entries, seed=seed)

    # Apply training regime (reorder/resample training data)
    train_entries = apply_training_regime(train_entries, training_regime, seed=seed)

    vocab = CharVocab()
    if use_phonological:
        all_seqs = [e["phon_src"] for e in entries] + [e["phon_tgt"] for e in entries]
    else:
        all_seqs = [list(e["orth_src"]) for e in entries] + [list(e["orth_tgt"]) for e in entries]
    vocab.build_from_sequences(all_seqs)

    train_ds = PastTenseDataset(train_entries, vocab, use_phonological,
                                use_edits=use_edits, use_edit_labels=use_edit_labels)
    val_ds = PastTenseDataset(val_entries, vocab, use_phonological,
                              use_edits=use_edits, use_edit_labels=use_edit_labels)
    test_ds = PastTenseDataset(test_entries, vocab, use_phonological,
                               use_edits=use_edits, use_edit_labels=use_edit_labels)
    return vocab, train_ds, val_ds, test_ds


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


def compute_accuracy(model, dataloader, vocab, device):
    """Compute exact-match accuracy via greedy decoding."""
    model.eval()
    correct = 0
    total = 0
    for src, tgt, *_ in dataloader:
        src, tgt = src.to(device), tgt.to(device)
        preds = model.greedy_decode(src, sos_idx=vocab.sos_idx, eos_idx=vocab.eos_idx)
        # Compare decoded sequences (strip <sos> and padding)
        for i in range(src.size(0)):
            pred_tokens = vocab.decode(preds[i].tolist())
            tgt_tokens = vocab.decode(tgt[i].tolist())
            if pred_tokens == tgt_tokens:
                correct += 1
            total += 1
    return correct / total if total > 0 else 0.0


def log(msg: str):
    """Print with immediate flush for piped output."""
    print(msg, flush=True)


def train(config: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")

    # Data
    regime = config.get("training_regime", "natural")
    use_edits = config.get("decoder_type") in ("edit", "population_edit")
    use_edit_labels = config.get("decoder_type") == "edit_labeler"
    vocab, train_ds, val_ds, test_ds = build_vocab_and_datasets(
        config["data_path"],
        use_phonological=config.get("use_phonological", True),
        seed=config.get("seed", 42),
        training_regime=regime,
        use_edits=use_edits,
        use_edit_labels=use_edit_labels,
    )
    log(f"Training regime: {regime}")
    pad_idx = vocab.pad_idx
    collate = partial(collate_fn, pad_idx=pad_idx)
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"],
                              shuffle=True, collate_fn=collate)
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
    ).to(device)

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
    epochs_without_improvement = 0

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

    # L0 annealing schedule: Phase 1 (all slots open) → Phase 2 (anneal to target)
    l0_anneal_start = config.get("l0_anneal_start", 0)  # epoch to begin annealing
    l0_anneal_epochs = config.get("l0_anneal_epochs", 0)  # epochs over which to anneal
    final_target_l0 = config.get("target_l0", 2.0)
    num_slots = config.get("num_slots", 8)

    for epoch in range(1, config["epochs"] + 1):
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

        for batch in train_loader:
            edit_tgt = None
            edit_lbl = None
            suffix_tgt = None
            if use_edit_labels:
                src, tgt, reg_labels, edit_lbl, suffix_tgt = batch
                edit_lbl = edit_lbl.to(device)
                suffix_tgt = suffix_tgt.to(device)
            elif use_edits:
                src, tgt, reg_labels, edit_tgt = batch
                edit_tgt = edit_tgt.to(device)
            else:
                src, tgt, reg_labels = batch
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
            result = model(src, tgt, src_for_recon=src_for_recon,
                           reg_labels=reg_labels, edit_targets=edit_tgt,
                           edit_labels=edit_lbl, suffix_targets=suffix_tgt)
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
                if use_edit_labels:
                    src, tgt, reg_labels, edit_lbl, suffix_tgt = batch
                    edit_lbl = edit_lbl.to(device)
                    suffix_tgt = suffix_tgt.to(device)
                elif use_edits:
                    src, tgt, reg_labels, edit_tgt = batch
                    edit_tgt = edit_tgt.to(device)
                else:
                    src, tgt, reg_labels = batch
                src, tgt = src.to(device), tgt.to(device)
                reg_labels = reg_labels.to(device)
                result = model(src, tgt, reg_labels=reg_labels, edit_targets=edit_tgt,
                               edit_labels=edit_lbl, suffix_targets=suffix_tgt)
                val_loss += result["loss"].item()
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

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "vocab": vocab,
                "config": config,
            }, save_dir / "best_model.pt")
        else:
            epochs_without_improvement += 1

        # Early stopping (disabled until after anneal begins, so model doesn't
        # stop before pruning has a chance to take effect)
        early_stop_after = l0_anneal_start if l0_anneal_start > 0 else 0
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
    log(f"Loaded best model from epoch {best_ckpt['epoch']} (val_loss={best_ckpt['val_loss']:.4f})")

    # Final test accuracy
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
