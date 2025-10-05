# ui_stump.py
import numpy as np
import pandas as pd
from stumpy import core, config
import numba
from numba import njit, prange
import ast


# -------------- mparray Class ----------------

class mparray(np.ndarray):
    """
    A matrix profile convenience class that subclasses numpy.ndarray.
    Provides .P_, .I_, .left_I_, .right_I_ for accessing profile results.
    """

    def __new__(cls, input_array, m, k, excl_zone_denom):
        obj = np.asarray(input_array).view(cls)
        obj._m = m
        obj._k = k
        obj._excl_zone_denom = excl_zone_denom
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._m = getattr(obj, "_m", None)
        self._k = getattr(obj, "_k", None)
        self._excl_zone_denom = getattr(obj, "_excl_zone_denom", None)

    def _P(self):
        if self._k == 1:
            return self[:, : self._k].flatten().astype(np.float64)
        else:
            return self[:, : self._k].astype(np.float64)

    def _I(self):
        if self._k == 1:
            return self[:, self._k : 2 * self._k].flatten().astype(np.int64)
        else:
            return self[:, self._k : 2 * self._k].astype(np.int64)

    def _left_I(self):
        if self._k == 1:
            return self[:, 2 * self._k].flatten().astype(np.int64)
        else:
            return self[:, 2 * self._k].astype(np.int64)

    def _right_I(self):
        if self._k == 1:
            return self[:, 2 * self._k + 1].flatten().astype(np.int64)
        else:
            return self[:, 2 * self._k + 1].astype(np.int64)

    @property
    def P_(self):
        return self._P()

    @property
    def I_(self):
        return self._I()

    @property
    def left_I_(self):
        return self._left_I()

    @property
    def right_I_(self):
        return self._right_I()


# ---------------- Encoding ----------------
def build_levels_table(paths, n):
    level_vocab = [dict() for _ in range(n)]

    def code(level, s):
        d = level_vocab[level]
        if s not in d:
            d[s] = len(d)
        return d[s]

    symbol_of_path = {}
    rows = []
    for p in paths:
        if len(p) != n:
            raise ValueError(f"Inconsistent path length: {len(p)} != {n}")
        if p not in symbol_of_path:
            sid = len(symbol_of_path)
            symbol_of_path[p] = sid
            rows.append([code(l, p[l]) for l in range(n)])
    levels = np.asarray(rows, dtype=np.int32)
    return symbol_of_path, levels

def encode_series(activity_paths, symbol_of_path):
    return np.array([symbol_of_path[p] for p in activity_paths], dtype=np.int32)

def hierarchy_weights(n):
    return np.array([2 ** (n - 1 - l) for l in range(n)], dtype=np.float64)


def build_cost_matrix(levels: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Efficient cost matrix computation with inclusive-level discount.

    Args:
        levels: (V, n) array of integer codes for each activity and each hierarchy level
        weights: (n,) array with weights ω_ℓ = 2^(n-ℓ)

    Returns:
        C: symmetric cost matrix (V, V)
    """
    V, n = levels.shape
    C = np.zeros((V, V), dtype=np.float64)

    for i in range(V):
        # Compare activity i to all others at once
        eq_matrix = (levels == levels[i])  # shape (V, n)

        # For each level ℓ, find mismatches
        diff = ~eq_matrix  # shape (V, n)

        # Precompute "equal lower levels" counts
        # eq_lower_counts[v, l] = # equal levels between v and i below l
        eq_lower_counts = np.zeros((V, n), dtype=np.int32)
        for l in range(n-1):
            eq_lower_counts[:, l] = eq_matrix[:, l+1:].sum(axis=1)

        # Calculate λ for mismatches
        lam = np.zeros((V, n), dtype=np.float64)
        for l in range(n):
            denom = (n - l)  # number of lower levels
            lam[:, l] = 1.0 - eq_lower_counts[:, l] / (denom + 1)

        # Contribution per level
        contrib = lam * weights * diff

        # Sum over levels → distances from i to all v
        dists = contrib.sum(axis=1)

        # Store into cost matrix
        C[i, :] = dists
        C[:, i] = dists  # maintain symmetry

    return C

@njit(fastmath=True)
def uihe_distance(a, b, weights):
    """
    Compute hierarchy-aware distance between two activities (Version A).
    
    Args:
        a, b: arrays of shape (n,), integer codes for each level
        weights: (n,) array with weights ωℓ = 2^(n-ℓ)
    """
    n = a.shape[0]
    d = 0.0
    for l in range(n):
        if a[l] != b[l]:
            # count equal lower levels
            eq_lower = 0
            for k in range(l+1, n):
                if a[k] == b[k]:
                    eq_lower += 1
            lam = 1.0 - eq_lower / ((n - l) + 1)
            d += lam * weights[l]
    return d

# ---------------- Core Distance Profile ----------------
@njit(fastmath=True)
def _ui_compute_diagonal(TA, TB, levels, weights, m, diags,
                         start, stop, thread_idx, k,
                         D, I, DL, DR, IL, IR, ignore_trivial):
    """
    Compute one diagonal of the hierarchy-aware (UIHE) distance matrix.

    This function is called from `_ui_stump()` to compute distances between
    all pairs of subsequences along a given diagonal offset `g` in the
    distance matrix of TA and TB. Each subsequence has length `m`.

    The distance between two subsequences is calculated using the
    hierarchy-aware symbolic distance `uihe_distance(a, b, weights)`, where
    each symbol `a` and `b` is represented by its hierarchical level vector.

    The results (top-k nearest neighbor distances and indices) are stored in
    the shared arrays D and I for the thread corresponding to `thread_idx`.

    Parameters
    ----------
    TA, TB : np.ndarray[int]
        Encoded symbolic sequences (activity or event IDs).
        TA is the reference (query) sequence, TB is the target.
    levels : np.ndarray[int]
        Symbol hierarchy table of shape (V, n_levels) where V = vocabulary size.
        Each row corresponds to a symbol and contains its hierarchy level encoding.
    weights : np.ndarray[float]
        Array of length n_levels specifying the weight for each level.
    m : int
        Subsequence length (window size).
    diags : np.ndarray[int]
        List of diagonal offsets to compute (distance matrix diagonals).
    start, stop : int
        Range of diagonals [start, stop) assigned to this thread.
    thread_idx : int
        Index of the current thread in parallel execution.
    k : int
        Number of nearest neighbors (top-k) to maintain for each subsequence.
    D : np.ndarray[float]
        3D array of per-thread distance profiles of shape (n_threads, l, k).
        Each row holds the top-k smallest distances for that subsequence.
    I : np.ndarray[int]
        3D array of per-thread index profiles of shape (n_threads, l, k).
        Stores the indices of the matching subsequences in TB.
    DL, DR : np.ndarray[float]
        Arrays storing the best (lowest-distance) left/right neighbor distances.
    IL, IR : np.ndarray[int]
        Arrays storing the corresponding indices for DL and DR.
    ignore_trivial : bool
        If True, skip trivial matches near the main diagonal (used for self-join).

    Notes
    -----
    - Each diagonal `g` corresponds to pairs (i, j = i + g).
      For g > 0 → upper diagonals, g < 0 → lower diagonals.
    - The computed distance is *average hierarchical cost* over m symbols.
    - Maintains sorted insertion for top-k motif distances via `np.searchsorted`.
    - Updates both sides (i, j) when `ignore_trivial=True`.
    """
    nA = TA.shape[0]
    m_inv = 1.0 / m

    # --- Iterate over assigned diagonals for this thread ---   
    for d_idx in range(start, stop):
        g = diags[d_idx]

        # Determine valid range of subsequence indices along this diagonal
        if g >= 0:
            rng = range(0, min(nA - m + 1, TB.shape[0] - m + 1 - g))
        else:
            rng = range(-g, min(nA - m + 1, TB.shape[0] - m + 1 - g))

        # --- Loop over subsequences along this diagonal ---
        for i in rng:
            j = i + g
            s = 0.0

            # --- Compute hierarchy-aware distance for subsequences [i:i+m] and [j:j+m] ---
            for t in range(m):
                a = levels[TA[i + t]]
                b = levels[TB[j + t]]
                # call on-the-fly distance
                s += uihe_distance(a, b, weights)
            dist = s * m_inv

            # --- Update the top-k profile for subsequence i ---
            rowD = D[thread_idx, i]
            rowI = I[thread_idx, i]
            if dist < rowD[-1]:
                idx = np.searchsorted(rowD, dist)
                rowD[idx+1:] = rowD[idx:-1]
                rowD[idx] = dist
                rowI[idx+1:] = rowI[idx:-1]
                rowI[idx] = j

            # --- Symmetric update for the matching subsequence j ---
            if ignore_trivial:
                rowDj = D[thread_idx, j]
                rowIj = I[thread_idx, j]
                if dist < rowDj[-1]:
                    idx = np.searchsorted(rowDj, dist)
                    rowDj[idx+1:] = rowDj[idx:-1]
                    rowDj[idx] = dist
                    rowIj[idx+1:] = rowIj[idx:-1]
                    rowIj[idx] = i

                # --- Update left/right nearest neighbor distances ---
                if i < j:
                    if dist < DL[thread_idx, j]:
                        DL[thread_idx, j] = dist
                        IL[thread_idx, j] = i
                    if dist < DR[thread_idx, i]:
                        DR[thread_idx, i] = dist
                        IR[thread_idx, i] = j


@njit(parallel=True, fastmath=True)
def _ui_stump(TA, TB, levels, weights, m, diags, ignore_trivial, k):
    """
    Compute the Hierarchy-Aware Matrix Profile (UI-STUMP) between two time series.

    This function performs a *parallelized hierarchical STUMP computation* that 
    generalizes the standard z-normalized Euclidean Matrix Profile to a 
    hierarchy-aware symbolic distance, defined by `levels` and `weights`.

    The algorithm iterates over diagonals in the pairwise distance matrix between
    subsequences of `TA` and `TB` of length `m`. Each diagonal is distributed
    across threads and processed using `_ui_compute_diagonal`.

    Parameters
    ----------
    TA : np.ndarray[int]
        Encoded symbolic time series A (length nA). Each value is an activity/symbol ID.
    TB : np.ndarray[int]
        Encoded symbolic time series B (length nB). Same symbolic encoding as TA.
    levels : np.ndarray[int]
        Array of shape (V, n_levels) describing the hierarchical encoding for each symbol.
    weights : np.ndarray[float]
        Array of shape (n_levels,) giving level-specific weights in the hierarchy-aware distance.
    m : int
        Window size (length of each subsequence to compare).
    diags : np.ndarray[int]
        Array of diagonal offsets to process (STUMP-style).
    ignore_trivial : bool
        If True, ignores trivial self-matches near the main diagonal (used in self-joins).
    k : int
        Number of top motif matches to keep per subsequence (top-k motifs).

    Returns
    -------
    D : np.ndarray[float]
        Distance matrix profile of shape (l, k), where l = nA - m + 1.
        Each row contains the k smallest distances for that subsequence (reversed for consistency).
    I : np.ndarray[int]
        Matrix of corresponding motif indices (shape (l, k)).
    IL : np.ndarray[int]
        Left nearest neighbor indices for each subsequence (shape (l,)).
    IR : np.ndarray[int]
        Right nearest neighbor indices for each subsequence (shape (l,)).

    Notes
    -----
    - This is a hierarchy-aware variant of STUMP that replaces correlation-based
      distances with UIHE symbolic distances computed in `_ui_compute_diagonal`.
    - Parallelization occurs across diagonal ranges, using Numba `prange`.
    - The merge phase combines partial thread results into global arrays.
    - The distance values are stored as negatives during merging for consistency
      with `_merge_topk_ρI` (which is optimized for maximizing correlation ρ).
    """
    # --- Initialize main constants and structures ---
    nA = TA.shape[0]
    l = nA - m + 1
    n_threads = numba.config.NUMBA_NUM_THREADS

    # --- Allocate per-thread storage for top-k distances and indices ---
    D = np.full((n_threads, l, k), np.inf)
    I = np.full((n_threads, l, k), -1)

    # Left/right nearest neighbor info per thread
    DL = np.full((n_threads, l), np.inf); IL = np.full((n_threads, l), -1)
    DR = np.full((n_threads, l), np.inf); IR = np.full((n_threads, l), -1)

    # --- Compute which diagonal segments each thread should process ---
    ndist_counts = core._count_diagonal_ndist(diags, m, TA.shape[0], TB.shape[0])
    diags_ranges = core._get_array_ranges(ndist_counts, n_threads, False)

    # --- Parallel computation of all diagonals ---
    for th in prange(n_threads):
        _ui_compute_diagonal(TA, TB, levels, weights, m, diags,
                             diags_ranges[th, 0], diags_ranges[th, 1],
                             th, k, D, I, DL, DR, IL, IR, ignore_trivial)

    # --- Merge per-thread results into global top-k profiles ---
    for th in range(1, n_threads):
        core._merge_topk_ρI(-D[0], -D[th], I[0], I[th])
        mask = DL[0] > DL[th]; DL[0][mask] = DL[th][mask]; IL[0][mask] = IL[th][mask]
        mask = DR[0] > DR[th]; DR[0][mask] = DR[th][mask]; IR[0][mask] = IR[th][mask]

    # --- Final formatting: reverse order for consistency with STUMP output ---
    D = D[0, :, ::-1]; I = I[0, :, ::-1]
    return D, I, IL[0], IR[0]


def stump(T_A_ids, levels, weights, m, T_B_ids=None, ignore_trivial=True, k=1):
    if T_B_ids is None:
        ignore_trivial = True
        T_B_ids = T_A_ids

    nA = T_A_ids.shape[0]; nB = T_B_ids.shape[0]
    core.check_window_size(m, max_size=min(nA, nB))
    excl = int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))
    if ignore_trivial:
        diags = np.arange(excl + 1, nA - m + 1, dtype=np.int64)
    else:
        diags = np.arange(-(nA - m + 1) + 1, nB - m + 1, dtype=np.int64)

    P, I, IL, IR = _ui_stump(T_A_ids, T_B_ids, levels, weights, m, diags, ignore_trivial, k)

    l = nA - m + 1
    out = np.empty((l, 2 * k + 2), dtype=object)
    out[:, :k] = P
    out[:, k:] = np.column_stack((I, IL, IR))
    return mparray(out, m, k, config.STUMPY_EXCL_ZONE_DENOM)


# ---------------- Helper for your dataframe ----------------
# --- robust value-to-string (handles NaN, lists, escapes separators) ---
def _ser(v):
    if pd.isna(v):
        return "__NA__"
    if isinstance(v, (list, tuple, set)):
        return "|".join(map(str, v))
    s = str(v)
    # escape our separators to keep tokens stable
    return s.replace("|", r"\|").replace("=", r"\=").replace("\n", r"\n")

def build_paths_from_df(df: pd.DataFrame, hierarchy_list, sep="|", kv_sep="="):
    """
    For each row, build a path of length len(hierarchy_list).
    Each level is serialized into ONE token by joining its columns.
    If all columns of a level are NA, the level token is '__NA__'.
    """
    paths = []
    cols_flat = [c for level in hierarchy_list for c in level]
    # optional sanity: ensure all referenced columns exist (warn only)
    missing = [c for c in cols_flat if c not in df.columns]
    if missing:
        # fill missing columns with NaN so code below stays uniform
        for c in missing:
            df[c] = np.nan

    for _, row in df.iterrows():
        level_tokens = []
        for level_cols in hierarchy_list:
            parts = []
            all_na = True
            for c in level_cols:
                val = row[c] if c in df.columns else np.nan
                sval = _ser(val)
                if sval != "__NA__":
                    all_na = False
                parts.append(f"{c}{kv_sep}{sval}")
            token = "__NA__" if all_na else sep.join(parts)
            level_tokens.append(token)
        paths.append(tuple(level_tokens))
    return paths

# ----- Discover Motifs -----
def discover_motifs(mp, top_k=5, window_size=30):
    """
    Discover top-k motifs using valley-based exclusion like STUMPY.
    """
    profile = np.array(mp.P_, dtype=float)
    indices = np.array(mp.I_, dtype=int)
    l = len(profile)

    discovered = []
    used = np.zeros(l, dtype=bool)

    for _ in range(top_k):
        # pick next minimum not masked
        mask = np.where(~used)[0]
        if len(mask) == 0:
            break
        i = mask[np.argmin(profile[mask])]
        j = indices[i]
        d = profile[i]

        discovered.append((i, j, d))

        # apply exclusion zone around both subsequences
        excl = window_size // 2
        lo_i, hi_i = max(0, i - excl), min(l, i + excl)
        lo_j, hi_j = max(0, j - excl), min(l, j + excl)
        used[lo_i:hi_i+1] = True
        used[lo_j:hi_j+1] = True

    return discovered

def compare_motifs(discovered, insertSpots, window_size=30, tolerance=None):
    """
    Compare discovered motifs (from ui_stump) against ground truth motif spots.
    
    Args:
        discovered: list of (i, j, dist) motif tuples
        insertSpots: ground truth motif indices (stringified list, list, or set)
        window_size: subsequence length
        tolerance: allowed deviation (defaults to window_size/2)

    Returns:
        insert_overlap: ground truth indices matched
        motif_overlap: discovered indices matched
        overlapDF: DataFrame with matches and alignmentAccuracy
    """
    if tolerance is None:
        tolerance = window_size // 2

    # --- normalize insertSpots into list[int] ---
    if isinstance(insertSpots, str):
        insertSpots = ast.literal_eval(insertSpots)
    if isinstance(insertSpots, (list, tuple, pd.Series, set)):
        flat = []
        for s in insertSpots:
            if isinstance(s, str):
                try:
                    vals = ast.literal_eval(s)
                    if isinstance(vals, (list, tuple)):
                        flat.extend([int(v) for v in vals])
                    else:
                        flat.append(int(vals))
                except Exception:
                    continue
            else:
                flat.append(int(s))
        gt_spots = set(flat)
    else:
        gt_spots = {int(insertSpots)}

    # --- discovered motif indices ---
    disc_spots = set([int(i) for i, j, d in discovered] + [int(j) for i, j, d in discovered])

    insert_overlap, motif_overlap = [], []
    rows = []

    for num1 in gt_spots:
        for num2 in disc_spots:
            if abs(num1 - num2) <= tolerance:
                insert_overlap.append(num1)
                motif_overlap.append(num2)
                rows.append({
                    "originalMotif": num1,
                    "discoveredMotif": num2,
                    "alignmentAccuracy": 1 - abs(num1 - num2) / tolerance
                })
                break  # avoid duplicates

    overlapDF = pd.DataFrame(rows)

    return insert_overlap, motif_overlap, overlapDF
