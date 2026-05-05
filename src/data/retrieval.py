"""Phonological-edit-distance kNN retrieval over training (src, tgt) pairs.

Used to bolt an exemplar-memory pathway onto the HMNT decoder: at decoding
time the model can attend not only over the source verb but also over the
past forms of the k nearest training verbs. Implements the "exemplar route"
of Pinker's dual-route theory of inflection.

Self-exclusion is by src-string equality: when querying with a training
item, all index entries with the same src are skipped (this also handles
oversampled-duplicate cases).
"""

from typing import List, Tuple
import sys


# -----------------------------------------------------------------------------
# Inflection-class taxonomy (auto-derivable from (src, tgt) alignment).
# Used by class-conditional retrieval: instead of edit-distance kNN over the
# whole train set (which surfaces surface-similar regulars for irregular
# queries), restrict kNN to items in the same predicted class so e.g. `deal`
# retrieves `keep/sleep/sweep` (the ept-class) rather than `pi:l/si:l/di:m`.
# -----------------------------------------------------------------------------
INFLECTION_CLASSES = [
    "REGULAR",     # 0  any regular suffix variant: +d, +t, +Id  (play→played)
    "NO_CHANGE",   # 1  identity past:            cut→cut
    "ABLAUT",      # 2  same-length vowel change: sing→sang
    "EPT",         # 3  vowel→E + final t/lt:     keep→kept, deal→dealt
    "OUGHT",       # 4  vowel→O: + t:             think→thought, buy→bought
    "OTE",         # 5  internal vowel→@U:        write→wrote, break→broke
    "OTHER_IRREG", # 6  catch-all (suppletive, complex):  go→went, take→took
]
# (Earlier v24a split regulars into REG_D/T/ID, which fragmented the
# regular pool into 3 smaller per-suffix classes. That reduced retrieval
# variety for regulars: a `walk` query couldn't see `play`/`want` neighbors
# under different suffixes. Collapsed into one REGULAR class so all regular
# queries can retrieve from the full ~3100-item pool.)
NUM_CLASSES = len(INFLECTION_CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(INFLECTION_CLASSES)}


def _alignment_ops(src_chars: List[str], tgt_chars: List[str]) -> List[str]:
    """Needleman-Wunsch alignment trace as a sequence of {COPY, SUB, INSERT, DELETE}.
    Language-agnostic — operates only on tokens, no morphology assumed.
    """
    n, m = len(src_chars), len(tgt_chars)
    INF = 10**9
    dp = [[INF] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 and j == 0:
                continue
            best = INF
            if i > 0:
                best = min(best, dp[i - 1][j] + 1)
            if j > 0:
                best = min(best, dp[i][j - 1] + 1)
            if i > 0 and j > 0:
                cost = 0 if src_chars[i - 1] == tgt_chars[j - 1] else 1
                best = min(best, dp[i - 1][j - 1] + cost)
            dp[i][j] = best

    ops: List[str] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if src_chars[i - 1] == tgt_chars[j - 1] else 1
            if dp[i][j] == dp[i - 1][j - 1] + cost:
                ops.append("COPY" if cost == 0 else "SUB")
                i -= 1
                j -= 1
                continue
        if j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            ops.append("INSERT")
            j -= 1
            continue
        ops.append("DELETE")
        i -= 1
    ops.reverse()
    return ops


def derive_pattern_signature(src_chars: List[str], tgt_chars: List[str]) -> tuple:
    """Unsupervised cluster signature for a (src, tgt) pair.

    Returns the *shape* of the alignment — the subsequence of non-COPY ops,
    so verbs that share a transformation pattern (e.g., all "vowel-change +
    suffix" verbs) collapse into the same signature regardless of stem
    length. Language-agnostic: makes no assumptions about morphology, just
    derives clusters from the alignment of (src, tgt) pairs.

    Examples (English past tense):
      play  -> played   ('INSERT', 'INSERT')           — pure suffix
      walk  -> walked   ('INSERT', 'INSERT')           — same cluster
      sing  -> sang     ('SUB',)                       — internal sub only
      meet  -> met      ('SUB',)                       — same cluster
      keep  -> kept     ('SUB', 'INSERT')              — sub + suffix
      sweep -> swept    ('SUB', 'INSERT')              — same cluster
      cut   -> cut      ()                             — identity
    """
    ops = _alignment_ops(src_chars, tgt_chars)
    return tuple(op for op in ops if op != "COPY")


def classify_inflection(src_chars: List[str], tgt_chars: List[str]) -> int:
    """Heuristic classification of a (src, tgt) past-tense pair into one of
    INFLECTION_CLASSES. Operates on phonological (DISC) char strings.
    """
    s = "".join(src_chars)
    t = "".join(tgt_chars)
    if s == t:
        return CLASS_TO_IDX["NO_CHANGE"]
    # K&C uses `r*` as a non-syllabic-r marker; in past tense the `r*`
    # is replaced by a single suffix consonant (E@r* -> E@d, dIlIv@r* -> dIlIv@d).
    # Treat these as regular variants.
    s_stripped = s[:-2] if s.endswith("r*") else s
    # Regular suffix patterns first (literal and r*-stripped variants).
    for suffix in ("d", "t", "Id"):
        if t == s + suffix or t == s_stripped + suffix:
            return CLASS_TO_IDX["REGULAR"]
    # Same-length: ablaut (sing/sang style).
    if len(s) == len(t):
        diffs = sum(1 for a, b in zip(s, t) if a != b)
        if 1 <= diffs <= 2:
            return CLASS_TO_IDX["ABLAUT"]
    # OUGHT: past contains O: + t (voiceless).
    if t.endswith("O:t") or "O:t" in t:
        return CLASS_TO_IDX["OUGHT"]
    # EPT: ends in -Et / -Elt / -Ept / -lt / -nt and vowel quality changed.
    if (t.endswith("Et") or t.endswith("Elt") or t.endswith("Ept")
            or t.endswith("lt") or t.endswith("nt")) and len(t) <= len(s) + 1:
        return CLASS_TO_IDX["EPT"]
    # OTE/OKE: past has @U vowel that wasn't in src (vowel→@U change).
    if "@U" in t and "@U" not in s:
        return CLASS_TO_IDX["OTE"]
    return CLASS_TO_IDX["OTHER_IRREG"]


def edit_distance(a, b) -> int:
    """Levenshtein distance between two sequences (lists/strings of hashables).
    O(len(a) * len(b)) with a one-row buffer.
    """
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[m]


class RetrievalIndex:
    """Precomputed kNN index over training (src_chars, tgt_chars) pairs.

    Tracks per-item inflection class labels (from `classify_inflection`) and
    supports class-conditional queries: `query_in_class(src, class_idx, k)`
    restricts the kNN search to items of a single class. This is what lets
    `deal` retrieve `keep/sleep/sweep` (the ept-class) instead of surface-
    similar but morphologically-irrelevant `pi:l/si:l/di:m`.

    Args:
        train_pairs: list of (src_chars, tgt_chars) tuples.
        k: default number of neighbors to return per query.
    """

    def __init__(self, train_pairs: List[Tuple[List[str], List[str]]], k: int = 5):
        self.train_pairs = train_pairs
        self.k = k
        self.n = len(train_pairs)
        # Map src tuple → list of train indices with that src (for self-exclusion)
        self._src_to_indices: dict = {}
        for i, (s, _) in enumerate(train_pairs):
            self._src_to_indices.setdefault(tuple(s), []).append(i)
        # Per-item inflection class + reverse index by class (English-specific)
        self.classes: List[int] = [classify_inflection(s, t) for s, t in train_pairs]
        self._class_to_indices: dict = {}
        for i, c in enumerate(self.classes):
            self._class_to_indices.setdefault(c, []).append(i)
        # Per-item unsupervised cluster signature + reverse index by cluster.
        # Language-agnostic: derived from alignment shape, not from any
        # hand-coded morphological taxonomy. Each unique signature gets an
        # integer ID; the registry is also exposed so the learned cluster
        # predictor head can target a fixed set of classes.
        self.cluster_signatures: List[tuple] = [
            derive_pattern_signature(s, t) for s, t in train_pairs
        ]
        self.cluster_registry: dict = {}
        for sig in self.cluster_signatures:
            if sig not in self.cluster_registry:
                self.cluster_registry[sig] = len(self.cluster_registry)
        self.cluster_ids: List[int] = [self.cluster_registry[sig]
                                       for sig in self.cluster_signatures]
        self.num_clusters: int = len(self.cluster_registry)
        self._cluster_to_indices: dict = {}
        for i, cid in enumerate(self.cluster_ids):
            self._cluster_to_indices.setdefault(cid, []).append(i)

    def _knn(self, src_chars: List[str], candidate_idxs: List[int],
             excluded: set, k: int) -> List[int]:
        distances = []
        for j in candidate_idxs:
            if j in excluded:
                continue
            d = edit_distance(src_chars, self.train_pairs[j][0])
            distances.append((d, j))
        distances.sort(key=lambda x: x[0])
        return [j for _, j in distances[:k]]

    def query(self, src_chars: List[str], k: int | None = None) -> List[int]:
        """Edit-distance kNN over the entire train set, excluding self."""
        excluded = set(self._src_to_indices.get(tuple(src_chars), []))
        return self._knn(src_chars, list(range(self.n)),
                         excluded, k or self.k)

    def query_random(self, src_chars: List[str], k: int | None = None,
                     rng=None) -> List[int]:
        """Negative control: sample k *random* train pairs, excluding self.

        If retrieval improves performance vs random retrieval, the *content*
        of retrieved exemplars matters (not just the presence of an extra
        attention pathway). This is the canonical ablation that rules out
        "retrieval just regularizes the model."
        """
        import random as _random
        rng = rng or _random.Random(hash(tuple(src_chars)) & 0xffffffff)
        excluded = set(self._src_to_indices.get(tuple(src_chars), []))
        candidates = [j for j in range(self.n) if j not in excluded]
        k = k or self.k
        if len(candidates) <= k:
            return candidates
        return rng.sample(candidates, k)

    def query_in_cluster(self, src_chars: List[str], cluster_id: int,
                         k: int | None = None) -> List[int]:
        """Edit-distance kNN restricted to items in a single unsupervised cluster.

        Same logic as `query_in_class` but uses auto-derived alignment-pattern
        clusters instead of the hand-coded English taxonomy. Falls back to a
        global query if the cluster is unknown (predicted ID out of range).
        """
        candidates = self._cluster_to_indices.get(cluster_id, [])
        if not candidates:
            return self.query(src_chars, k=k)
        excluded = set(self._src_to_indices.get(tuple(src_chars), []))
        return self._knn(src_chars, candidates, excluded, k or self.k)

    def cluster_id_for(self, src_chars: List[str], tgt_chars: List[str]) -> int:
        """True cluster ID for a (src, tgt) pair (used at training time)."""
        sig = derive_pattern_signature(src_chars, tgt_chars)
        return self.cluster_registry.get(sig, -1)

    def query_in_class(self, src_chars: List[str], class_idx: int,
                       k: int | None = None) -> List[int]:
        """Edit-distance kNN restricted to items of a single class.

        If the class has fewer than k members, returns however many exist
        (sorted by distance). Falls back to a global query if the class is
        empty (shouldn't happen in practice if class_idx came from the
        learned predictor over training-class probabilities).
        """
        candidates = self._class_to_indices.get(class_idx, [])
        if not candidates:
            return self.query(src_chars, k=k)
        excluded = set(self._src_to_indices.get(tuple(src_chars), []))
        return self._knn(src_chars, candidates, excluded, k or self.k)

    def get_targets(self, indices: List[int]) -> List[List[str]]:
        return [self.train_pairs[i][1] for i in indices]


# --- self-test ---
if __name__ == "__main__":
    pairs = [
        (list("play"),   list("played")),
        (list("ring"),   list("rang")),
        (list("sing"),   list("sang")),
        (list("spring"), list("sprang")),
        (list("walk"),   list("walked")),
        (list("talk"),   list("talked")),
        (list("keep"),   list("kept")),
        (list("sleep"),  list("slept")),
        (list("sweep"),  list("swept")),
        (list("go"),     list("went")),
    ]
    idx = RetrievalIndex(pairs, k=3)

    # Query with a training item — should exclude self
    top = idx.query(list("ring"))
    targets = idx.get_targets(top)
    print(f"Query 'ring' -> top-{idx.k}:")
    for j, t in zip(top, targets):
        src = "".join(pairs[j][0])
        tgt = "".join(t)
        print(f"  idx={j:2d}  src={src:8s}  past={tgt}")

    # Query with a "wug" verb (not in train)
    top = idx.query(list("dwell"))
    targets = idx.get_targets(top)
    print(f"\nQuery 'dwell' -> top-{idx.k}:")
    for j, t in zip(top, targets):
        src = "".join(pairs[j][0])
        tgt = "".join(t)
        print(f"  idx={j:2d}  src={src:8s}  past={tgt}")

    # Verify oversample-style duplicate is also excluded
    pairs2 = pairs + [(list("ring"), list("rang"))]  # add a duplicate
    idx2 = RetrievalIndex(pairs2, k=3)
    top = idx2.query(list("ring"))
    print(f"\nWith duplicate, query 'ring' -> top-{idx2.k} (should not include 'ring'):")
    for j in top:
        print(f"  idx={j:2d}  src={''.join(pairs2[j][0])}")
    assert all(pairs2[j][0] != list("ring") for j in top), "Self-exclusion failed!"
    print("Self-exclusion OK")
