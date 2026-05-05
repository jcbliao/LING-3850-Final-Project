"""Hard Monotonic Neural Transducer (HMNT) action vocabulary, oracle, and aligner.

Action vocab layout (size = 3 + char_vocab_size):
    0       = ACTION_PAD   (ignored in loss)
    1       = ACTION_STEP  (advance source pointer by 1)
    2       = ACTION_END   (terminate output)
    3 + c   = WRITE(c)     (append character c to output, pointer unchanged)

Action ordering convention (from align_to_actions):
    COPY of src[i] (matches tgt[j]) -> [WRITE(tgt[j]), STEP]
    SUB  of src[i] -> tgt[j]        -> [WRITE(tgt[j]), STEP]
    INSERT tgt[j]                   -> [WRITE(tgt[j])]
    DELETE src[i]                   -> [STEP]

Reference: Aharoni & Goldberg 2017 "Morphological Inflection Generation
with Hard Monotonic Attention" (simpler variant of their action set).
"""

from typing import List


ACTION_PAD = 0
ACTION_STEP = 1
ACTION_END = 2
WRITE_OFFSET = 3


def action_vocab_size(char_vocab_size: int) -> int:
    return WRITE_OFFSET + char_vocab_size


def write_id(char_idx: int) -> int:
    return WRITE_OFFSET + char_idx


def is_write(action_id: int) -> bool:
    return action_id >= WRITE_OFFSET


def write_char(action_id: int) -> int:
    return action_id - WRITE_OFFSET


def _nw_dp(src: List[int], tgt: List[int]) -> List[List[int]]:
    """Needleman-Wunsch with unit costs (match=0, sub/ins/del=1)."""
    n, m = len(src), len(tgt)
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
                cost = 0 if src[i - 1] == tgt[j - 1] else 1
                best = min(best, dp[i - 1][j - 1] + cost)
            dp[i][j] = best
    return dp


def align_to_actions(src_ids: List[int], tgt_ids: List[int]) -> List[int]:
    """Convert (src, tgt) into a deterministic HMNT action script ending with END."""
    n, m = len(src_ids), len(tgt_ids)
    dp = _nw_dp(src_ids, tgt_ids)

    # Traceback (right-to-left); then reverse so WRITE precedes STEP for COPY/SUB.
    actions: List[int] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if src_ids[i - 1] == tgt_ids[j - 1] else 1
            if dp[i][j] == dp[i - 1][j - 1] + cost:
                actions.append(ACTION_STEP)
                actions.append(write_id(tgt_ids[j - 1]))
                i -= 1
                j -= 1
                continue
        if j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            actions.append(write_id(tgt_ids[j - 1]))
            j -= 1
            continue
        # DELETE
        actions.append(ACTION_STEP)
        i -= 1
    actions.reverse()
    actions.append(ACTION_END)
    return actions


def apply_actions(src_ids: List[int], action_ids: List[int]) -> List[int]:
    """Deterministically apply an action sequence to source characters."""
    n = len(src_ids)
    output: List[int] = []
    ptr = 0
    for a in action_ids:
        if a == ACTION_END:
            break
        elif a == ACTION_STEP:
            ptr = min(ptr + 1, n)  # safety: don't advance past end
        elif is_write(a):
            output.append(write_char(a))
        # ACTION_PAD: skip
    return output


def oracle_next_action(src_ids: List[int], tgt_ids: List[int],
                       ptr: int, output_len: int) -> int:
    """Optimal next action given the current state.

    Assumes the output produced so far equals tgt_ids[:output_len]. Returns
    the first action of `align_to_actions(src[ptr:], tgt[output_len:])` so
    the tie-breaking convention is identical to the precomputed dataset
    targets — critical for DAgger, where on-script samples must reuse the
    precomputed targets without distribution drift on tie-broken alignments.
    """
    n = len(src_ids) - ptr
    m = len(tgt_ids) - output_len
    if m == 0:
        return ACTION_STEP if n > 0 else ACTION_END
    if n == 0:
        return write_id(tgt_ids[output_len])
    actions = align_to_actions(src_ids[ptr:], tgt_ids[output_len:])
    return actions[0] if actions else ACTION_END


# --- self-test ---
if __name__ == "__main__":
    import random

    def _check_roundtrip(src, tgt):
        actions = align_to_actions(src, tgt)
        out = apply_actions(src, actions)
        assert out == tgt, f"Roundtrip failed: src={src} tgt={tgt} actions={actions} out={out}"

    # Hand-crafted cases
    cases = [
        ([0, 1, 2, 3], [0, 1, 2, 3, 4, 5]),  # copy + 2 inserts
        ([0, 1, 2], [3, 4, 5]),               # full sub
        ([0, 1, 2, 3], [0, 5, 2, 3]),         # vowel change
        ([], [1, 2]),                         # empty src
        ([1, 2], []),                         # empty tgt
        ([1], [1]),                           # identity
        ([0, 1, 2], [0, 1, 2]),               # identity longer
    ]
    for src, tgt in cases:
        _check_roundtrip(src, tgt)
    print(f"Roundtrip cases: {len(cases)} passed")

    # Random fuzz
    rng = random.Random(0)
    for _ in range(200):
        n = rng.randint(0, 8)
        m = rng.randint(0, 8)
        src = [rng.randint(0, 5) for _ in range(n)]
        tgt = [rng.randint(0, 5) for _ in range(m)]
        _check_roundtrip(src, tgt)
    print("Random fuzz: 200 passed")

    # Oracle equivalence: walking from (0,0) with the oracle should reproduce target
    def _oracle_rollout(src, tgt, max_steps=64):
        ptr = 0
        out_len = 0
        out: List[int] = []
        for _ in range(max_steps):
            a = oracle_next_action(src, tgt, ptr, out_len)
            if a == ACTION_END:
                break
            elif a == ACTION_STEP:
                ptr = min(ptr + 1, len(src))
            elif is_write(a):
                out.append(write_char(a))
                out_len += 1
        return out

    for _ in range(200):
        n = rng.randint(0, 8)
        m = rng.randint(0, 8)
        src = [rng.randint(0, 5) for _ in range(n)]
        tgt = [rng.randint(0, 5) for _ in range(m)]
        out = _oracle_rollout(src, tgt)
        assert out == tgt, f"Oracle rollout failed: src={src} tgt={tgt} out={out}"
    print("Oracle rollout: 200 passed")

    # Inspect a morphology-style example
    # play -> played: [p,l,a,y] -> [p,l,a,y,e,d]
    src = [10, 11, 12, 13]
    tgt = [10, 11, 12, 13, 14, 15]
    actions = align_to_actions(src, tgt)
    print(f"play->played actions: {actions}")
    print(f"  decoded: {[ 'STEP' if a==ACTION_STEP else 'END' if a==ACTION_END else f'W({write_char(a)})' for a in actions]}")
