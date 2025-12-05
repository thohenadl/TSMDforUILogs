# ----- SKIMP Implementation -----
import numpy as np
import time
from typing import Iterable, List, Tuple, Optional
from util.ui_stump import stump, mparray  # your file

# ---------- 1) Balanced-BFS order for lengths ----------
def _balanced_bfs_lengths(L: int, U: int, S: int = 1) -> List[int]:
    """
    Build a balanced-BFS order of lengths in [L, U] with step S.
    Produces a breadth-first subdivision of the range (SKIMP, Table 2).
    """
    # Canonical discrete grid
    grid = list(range(L, U + 1, S))
    if not grid:
        return []
    # BFS over segments: start from the whole interval, repeatedly split
    queue = [(0, len(grid) - 1)]  # indices in grid
    order_idx = []
    while queue:
        lo, hi = queue.pop(0)
        mid = (lo + hi) // 2
        order_idx.append(mid)
        # left segment
        if lo <= mid - 1:
            queue.append((lo, mid - 1))
        # right segment
        if mid + 1 <= hi:
            queue.append((mid + 1, hi))
    return [grid[i] for i in order_idx]

# ---------- 2) Storage helpers ----------
class PMPStore:
    """
    Compact PMP holder. Keeps MPs as float32; indices optional.
    Can downsample in width to save RAM (only needed for visualization).
    """
    def __init__(self, n_rows:int, row_len:int, keep_indices:bool=True, dtype=np.float32):
        self.P = np.full((n_rows, row_len), np.inf, dtype=dtype)
        self.I = np.full((n_rows, row_len), -1, dtype=np.int32) if keep_indices else None
        self.lengths: List[int] = []

    def add_row(self, row_idx:int, mp: mparray, m: int):
        # mp.P_ and mp.I_ are length (row_len = |T|-m+1)
        self.P[row_idx, :] = mp.P_.astype(self.P.dtype, copy=False)
        if self.I is not None:
            self.I[row_idx, :] = mp.I_.astype(np.int32, copy=False)
        self.lengths.append(m)

# ---------- 3) Simple multi-length motif aggregation ----------
def _dedup_motifs_across_lengths(cands: List[Tuple[int,int,int,float,float]], min_sep:int) -> List[Tuple[int,int,int,float,float]]:
    """
    Deduplicate motif candidates across lengths by index overlap.
    Keep longest representative first.
    Each cand: (i, j, m, d, score).
    """
    cands = sorted(cands, key=lambda x: (-x[4], x[3]))  # consistent ordering
    kept = []
    used = np.zeros(len(cands), dtype=bool)
    for a in range(len(cands)):
        if used[a]:
            continue
        ia, ja, ma = cands[a][0], cands[a][1], cands[a][2]
        kept.append(cands[a])
        for b in range(a + 1, len(cands)):
            if used[b]:
                continue
            ib, jb, mb = cands[b][0], cands[b][1], cands[b][2]
            if (abs(ia - ib) < max(min_sep, ma // 2)) or (abs(ja - jb) < max(min_sep, ma // 2)):
                used[b] = True
    return kept


# ---------- 4) SKIMP driver ----------
def skimp_uihe(
    series_ids: np.ndarray,      # shape (N,), int32 ids into levels table
    levels: np.ndarray,          # shape (V, n), codes for UIHE levels
    weights: np.ndarray,         # shape (n,), ωℓ = 2^(n-ℓ)
    L: int, U: int, S: int = 1,  # length range
    k: int = 1,                  # top-k profile per index (as in stump)
    ignore_trivial: bool = True,
    max_rows: Optional[int] = None,     # anytime budget: max #lengths to compute
    time_budget_s: Optional[float] = None,  # anytime budget: max wall clock seconds
    keep_indices: bool = True,
    return_pmp: bool = True,
    motif_topK: int = 10,               # cross-length motif mining
    min_separation: Optional[int] = None,   # cross-length motif dedup
) -> dict:
    """
    SKIMP-style Pan Matrix Profile for UIHE distance (anytime).
    - Balanced-BFS length ordering (Table 2 in paper).
    - Calls ui_stump.stump for each sampled length (your on-the-fly distance).
    - Returns PMP (float32) + aggregated cross-length motif candidates.
    """
    N = series_ids.shape[0]
    # fast sanity
    if L < 2 or U < L:
        raise ValueError("Invalid length bounds")
    # Balanced BFS order over lengths
    lengths = _balanced_bfs_lengths(L, U, S)

    # Determine storage width = |T|-m+1 for the smallest m
    min_m = min(lengths)
    row_len = N - min_m + 1
    if row_len <= 0:
        raise ValueError("Window too long for series")

    # Prepare PMP store
    n_rows = len(lengths) if max_rows is None else min(len(lengths), max_rows)
    store = PMPStore(n_rows=n_rows, row_len=row_len, keep_indices=keep_indices, dtype=np.float32)

    # Anytime loop
    t0 = time.time()
    computed = 0
    motif_pool: List[Tuple[int,int,int,float]] = []  # (i, j, m, d)

    for r_idx, m in enumerate(lengths):
        if max_rows is not None and computed >= max_rows:
            break
        if time_budget_s is not None and (time.time() - t0) > time_budget_s:
            break

        # For each m, we need a profile of length (N - m + 1):
        # For consistency across rows, we will right-pad with inf when m > min_m
        row_len_m = N - m + 1
        if row_len_m <= 0:
            continue

        # Compute MP for this m using your UIHE stump (already self-join safe via exclusion zone)
        mp = stump(series_ids, levels, weights, m=m, T_B_ids=None, ignore_trivial=ignore_trivial, k=k)
        # After computing mp for length m
        row_len_m = N - m + 1
        P_m = mp.P_.astype(np.float32, copy=False)[:row_len_m]
        I_m = mp.I_.astype(np.int32, copy=False)[:row_len_m]


        # Insert into PMP storage (align right so columns roughly compare; or left, but keep consistent)
        # Here: left-align. If row_len_m < row_len, we pad the tail with inf to same width.
        # Copy to temp full-length row
        fullP = np.full(row_len, np.inf, dtype=np.float32)
        fullI = np.full(row_len, -1, dtype=np.int32)
        fullP[:row_len_m], fullI[:row_len_m] = P_m, I_m

        # Save
        # We wrap P/I back into an mparray-shaped container only for storage uniformity
        # (store expects mp-like getters; we can directly set arrays instead)
        store.P[r_idx, :], store.I[r_idx, :] = fullP, fullI
        store.lengths.append(m)
        computed += 1

        # Extract a few motif candidates at this m to build cross-length pool
        # Greedy minima with larger exclusion to avoid near-duplicates
        prof = fullP
        # ---- Candidate extraction for this m (NO padding used here) ----
        idxs = np.arange(row_len_m, dtype=int)
        used = np.zeros(row_len_m, dtype=bool)
        excl = max(5, m // 2)

        # Precompute a percentile score for this length
        # Stable tie-breaking: (score desc, distance asc)
        ranks = np.argsort(np.argsort(P_m))        # 0..row_len_m-1
        score = 1.0 - ranks / max(1, (row_len_m - 1))  # in [0,1]

        for _ in range(min(motif_topK, row_len_m)):
            mask = np.where(~used)[0]
            if len(mask) == 0:
                break
            # Choose by percentile score (desc), then distance (asc)
            best = max(mask, key=lambda i: (score[i], -P_m[i]))
            ii, jj, d = int(best), int(I_m[best]), float(P_m[best])
            if jj < 0 or jj >= row_len_m:
                used[ii] = True
                continue
            # Use 'score' instead of rho for cross-length merge
            motif_pool.append((ii, jj, int(m), d, float(score[ii])))
            lo_i, hi_i = max(0, ii - excl), min(row_len_m - 1, ii + excl)
            lo_j, hi_j = max(0, jj - excl), min(row_len_m - 1, jj + excl)
            used[lo_i:hi_i+1] = True
            used[lo_j:hi_j+1] = True

    # Cross-length motif aggregation & dedup
    if min_separation is None:
        # either the shortest expected motif length, or the step size
        min_separation = min(S, L)

    # Normalize rho within each length before pooling
    for m in lengths:
        mask = [cand for cand in motif_pool if cand[2] == m]
        if len(mask) > 1:
            rhos = np.array([c[4] for c in mask])
            mu, sigma = np.mean(rhos), np.std(rhos)
            for k, c in enumerate(mask):
                idx = motif_pool.index(c)
                rho_norm = (c[4] - mu) / (sigma if sigma > 0 else 1)
                motif_pool[idx] = (*c[:4], rho_norm)

    # Rank by correlation (descending), then by distance (ascending)
    motif_pool = sorted(motif_pool, key=lambda x: (-x[4], x[3]))
    motifs_agg = _dedup_motifs_across_lengths(motif_pool, min_sep=min_separation or L)[:motif_topK]

    def uihe_pair_distance(series_ids, levels, weights, i, j, m) -> float:
    # TODO: Implement the UIHE distance between two subsequences [i:i+m), [j:j+m)
    # e.g., sum over level weights * cost(levels[ids[start..start+m]])
    # Placeholder:
        raise NotImplementedError

    # --- Local refinement to correct index mismatches (±2 around motif start) ---
    REFINE_DELTA = 2
    refined = []
    for (i, j, m, d, rho) in motifs_agg:
        best = (i, j, m, d, rho)
        # Re-evaluate nearby indices (±2) to re-align the motif start
        for shift in range(-REFINE_DELTA, REFINE_DELTA + 1):
            i_shift = max(0, i + shift)
            if i_shift + m >= len(levels):
                continue
            # Compute the distance again at this shifted position
            mp_segment = stump(series_ids, levels, weights, m=m, ignore_trivial=True, k=1)
            d2 = float(mp_segment.P_[i_shift])
            rho2 = 1.0 - (d2 * d2) / (2.0 * m)
            if rho2 > best[4] or (rho2 == best[4] and d2 < best[3]):
                best = (i_shift, int(mp_segment.I_[i_shift]), m, d2, rho2)
        refined.append(best)

    motifs_agg = refined

    out = {
        "lengths": store.lengths[:computed],
        "PMP": store.P[:computed, :] if return_pmp else None,
        "PMPI": store.I[:computed, :] if (return_pmp and store.I is not None) else None,
        "motifs": motifs_agg,   # list of (i, j, m, d, rho)
        "computed_rows": computed,
        "row_width": row_len,
        "anytime_seconds": time.time() - t0,
    }
    return out