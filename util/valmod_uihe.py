# File generated/modified with Codex assistance on 2025-10-09
# Reviewed and approved by: tomho
"""
VALMOD implementation specialized for the UI hierarchy-aware distance measure.

This module adapts the Variable-Length Motif Discovery (VALMOD) workflow from
Linardi et al. (2020), "Matrix Profile Goes MAD: Variable-Length Motif and
Discord Discovery in Data Series", Algorithms 1-6. All computations are carried
out with the hierarchy-aware symbolic distance `uihe_distance`, ensuring that
motif discovery respects UI taxonomy constraints. The emphasis is on clarity
and reproducibility rather than raw performance.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


__all__ = [
    "uihe_distance",
    "subsequence_distance",
    "compute_matrix_profile",
    "compute_sub_matrix_profile",
    "update_valmp",
    "valmod",
]


def uihe_distance(activity_a: Sequence[int], activity_b: Sequence[int], weights: np.ndarray) -> float:
    """
    Compute the hierarchy-aware event distance between two encoded activities.

    This is the hierarchy-aware alternative to Euclidean distance described in
    Algorithm 3 of Linardi et al. (2020) when adapted to symbolic hierarchies.
    The implementation mirrors `util.ui_stump.uihe_distance` but is included
    locally for clarity and traceability.

    Parameters
    ----------
    activity_a, activity_b : Sequence[int]
        Integer codes for each hierarchy level.
    weights : np.ndarray
        Weight vector where ``weights[ell] = 2**(n_levels-ell-1)`` following the
        UIHE weighting scheme.

    Returns
    -------
    float
        Hierarchy-aware dissimilarity between the activities.
    """
    if weights is None:
        raise ValueError("weights must be provided for hierarchy-aware distance computation.")
    if len(activity_a) != len(activity_b):
        raise ValueError("Activities must align across hierarchy levels.")

    mismatch_cost = 0.0
    n_levels = len(activity_a)
    for level_idx in range(n_levels):
        if activity_a[level_idx] != activity_b[level_idx]:
            equal_suffix = 0
            for lower_idx in range(level_idx + 1, n_levels):
                if activity_a[lower_idx] == activity_b[lower_idx]:
                    equal_suffix += 1
            lam = 1.0 - equal_suffix / ((n_levels - level_idx) + 1)
            mismatch_cost += lam * weights[level_idx]
    return mismatch_cost


def subsequence_distance(window_a: np.ndarray, window_b: np.ndarray, weights: np.ndarray) -> float:
    """
    Average UI hierarchy-aware distance between two subsequences.

    Mirrors Algorithm 4 (ComputeSubMP) by aggregating per-event distances. Each
    event distance leverages `uihe_distance`, and the aggregate is normalized by
    the subsequence length to maintain comparability across lengths.

    Parameters
    ----------
    window_a, window_b : np.ndarray
        Subsequence views of shape (m, n_levels).
    weights : np.ndarray
        Hierarchy weights passed to `uihe_distance`.

    Returns
    -------
    float
        Mean hierarchy-aware mismatch cost across aligned events.
    """
    if window_a.shape != window_b.shape:
        raise ValueError("Subsequences must have identical shape for comparison.")
    per_step = np.fromiter(
        (uihe_distance(window_a[idx], window_b[idx], weights) for idx in range(window_a.shape[0])),
        dtype=np.float64,
        count=window_a.shape[0],
    )
    return float(per_step.mean())


def _resolve_activity_matrix(series_ids: Sequence[int], levels: np.ndarray) -> np.ndarray:
    """
    Expand a 1D sequence of symbol identifiers into their hierarchy vectors.

    This mirrors the data access pattern used by `util.ui_stump.stump`, ensuring
    the VALMOD driver consumes the same symbolic model without requiring the
    caller to pre-materialize activity vectors.

    Parameters
    ----------
    series_ids : Sequence[int]
        Encoded activity identifiers of shape (N,).
    levels : np.ndarray
        Lookup table of shape (V, n_levels) storing hierarchy codes.

    Returns
    -------
    np.ndarray
        Activity matrix of shape (N, n_levels) obtained via `levels[series_ids]`.
    """
    series_ids_arr = np.asarray(series_ids, dtype=np.int64)
    if series_ids_arr.ndim != 1:
        raise ValueError("series_ids must be a one-dimensional array of integer identifiers.")
    if levels.ndim != 2:
        raise ValueError("levels must be a two-dimensional lookup table.")
    max_id = series_ids_arr.max(initial=-1)
    if max_id >= levels.shape[0]:
        raise ValueError("series_ids contain indices outside the provided levels table.")
    if np.any(series_ids_arr < 0):
        raise ValueError("series_ids must be non-negative.")
    return levels[series_ids_arr]


def _extract_subsequences(series: np.ndarray, window: int) -> np.ndarray:
    """
    Construct all contiguous subsequences of given length from the series.

    Parameters
    ----------
    series : np.ndarray
        Array of shape (N, n_levels).
    window : int
        Subsequence length.

    Returns
    -------
    np.ndarray
        Array of shape (N - window + 1, window, n_levels) holding subsequences.
    """
    n_samples = series.shape[0] - window + 1
    if n_samples <= 0:
        raise ValueError("Window longer than the series.")
    return np.stack([series[start : start + window] for start in range(n_samples)], axis=0)


def _compute_distance_matrix(subseqs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Build the full pairwise distance matrix for the provided subsequences.

    Parameters
    ----------
    subseqs : np.ndarray
        Array of shape (n_subseq, window, n_levels).
    weights : np.ndarray
        Hierarchy weights vector.

    Returns
    -------
    np.ndarray
        Symmetric matrix of shape (n_subseq, n_subseq) with UIHE distances.
    """
    n_subseq = subseqs.shape[0]
    dist_matrix = np.full((n_subseq, n_subseq), np.inf, dtype=np.float64)
    for i in range(n_subseq):
        dist_matrix[i, i] = 0.0
        for j in range(i + 1, n_subseq):
            dist_val = subsequence_distance(subseqs[i], subseqs[j], weights)
            dist_matrix[i, j] = dist_val
            dist_matrix[j, i] = dist_val
    return dist_matrix


def compute_matrix_profile(
    activity_matrix: Sequence[Sequence[int]],
    subseq_length: int,
    p: int,
    weights: np.ndarray,
    exclusion_fraction: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the UIHE matrix profile for a fixed subsequence length.

    Implements Algorithm 2 (VALMOD initialization) using hierarchy-aware
    distances. The algorithm forms the distance matrix, enforces a self-join
    exclusion zone, and extracts the nearest neighbor (motif) distances.

    Parameters
    ----------
    activity_matrix : Sequence[Sequence[int]]
        Fully expanded activity stream of shape (N, n_levels).
    subseq_length : int
        Window length (symbol count) for the subsequences.
    p : int
        Number of top motifs to retain for downstream aggregation.
    weights : np.ndarray
        Hierarchy weights aligned with the second dimension of `activity_matrix`.
    exclusion_fraction : float, optional
        Fraction of the window size used for the self-join exclusion zone.

    Returns
    -------
    matrix_profile : np.ndarray
        Array of size (N - subseq_length + 1,) with nearest neighbor distances.
    profile_indices : np.ndarray
        Matching indices for the matrix profile values.
    distance_matrix : np.ndarray
        Symmetric UIHE distance matrix for reuse in incremental steps.
    """
    series = np.asarray(activity_matrix, dtype=np.int64)
    if series.ndim != 2:
        raise ValueError("activity_matrix must be two-dimensional: (time, hierarchy_levels).")
    if subseq_length < 2:
        raise ValueError("subseq_length must be at least 2.")
    if subseq_length > series.shape[0]:
        raise ValueError("subseq_length cannot exceed the series length.")
    if p < 1:
        raise ValueError("p must be positive.")

    subseqs = _extract_subsequences(series, subseq_length)
    dist_matrix = _compute_distance_matrix(subseqs, weights)

    n_subseq = subseqs.shape[0]
    exclusion = max(1, int(np.floor(exclusion_fraction * subseq_length)))

    matrix_profile = np.full(n_subseq, np.inf, dtype=np.float64)
    profile_indices = np.full(n_subseq, -1, dtype=np.int64)

    for idx in range(n_subseq):
        distances = dist_matrix[idx].copy()
        start_exc = max(0, idx - exclusion)
        stop_exc = min(n_subseq, idx + exclusion + 1)
        distances[start_exc:stop_exc] = np.inf

        best_match = np.argmin(distances)
        best_distance = distances[best_match]
        if not np.isfinite(best_distance):
            continue

        matrix_profile[idx] = best_distance
        profile_indices[idx] = int(best_match)

    return matrix_profile, profile_indices, dist_matrix


def compute_sub_matrix_profile(
    activity_matrix: Sequence[Sequence[int]],
    n_dp: int,
    listDP: Optional[Dict[int, np.ndarray]],
    new_length: int,
    p: int,
    weights: np.ndarray,
    exclusion_fraction: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, np.ndarray], np.ndarray]:
    """
    Update the VALMOD structure with a new subsequence length.

    Algorithm 4 (ComputeSubMatrixProfile) recomputes the matrix profile for a
    new length and retains the leading distance profiles for re-use. Here we
    adopt a transparent implementation that stores the top-n distance rows.

    Parameters
    ----------
    activity_matrix : Sequence[Sequence[int]]
        Expanded hierarchy-aware activity matrix (N, n_levels).
    n_dp : int
        Number of distance profiles to preserve for reuse (VALMOD's DP list).
    listDP : dict[int, np.ndarray] or None
        Cache mapping lengths to stored distance profiles.
    new_length : int
        Subsequence length requested by Algorithm 1's scheduler.
    p : int
        Number of motifs retained for the VALMP aggregator.
    weights : np.ndarray
        Hierarchy weights compatible with `T`.
    exclusion_fraction : float, optional
        Exclusion window fraction for trivial matches.

    Returns
    -------
    matrix_profile : np.ndarray
        Updated matrix profile for `new_length`.
    profile_indices : np.ndarray
        Companion index array for the matrix profile.
    listDP : dict[int, np.ndarray]
        Updated cache with the strongest `n_dp` distance profiles.
    distance_matrix : np.ndarray
        The raw pairwise distance matrix for further analysis if required.
    """
    mp, mpi, distance_matrix = compute_matrix_profile(
        activity_matrix=activity_matrix,
        subseq_length=new_length,
        p=p,
        weights=weights,
        exclusion_fraction=exclusion_fraction,
    )

    if listDP is None:
        listDP = {}

    top_order = np.argsort(mp)
    keep = min(n_dp, top_order.size)
    selected_profiles = distance_matrix[top_order[:keep]]
    listDP[new_length] = selected_profiles

    return mp, mpi, listDP, distance_matrix


@dataclass
class VALMODState:
    """
    Container maintaining VALMOD bookkeeping during Algorithm 1 iterations.

    Attributes
    ----------
    distances : Dict[int, np.ndarray]
        Map from subsequence length to matrix profile distances.
    indices : Dict[int, np.ndarray]
        Map from subsequence length to nearest neighbor indices.
    motifs : Dict[Tuple[int, int, int], Dict[str, float]]
        Motif dictionary keyed by (length, start, match).
    lengths : List[int]
        Ordered list of processed lengths.
    """

    distances: Dict[int, np.ndarray]
    indices: Dict[int, np.ndarray]
    motifs: Dict[Tuple[int, int, int], Dict[str, float]]
    lengths: List[int]


def update_valmp(
    state: VALMODState,
    length: int,
    matrix_profile: np.ndarray,
    profile_indices: np.ndarray,
    p: int,
) -> VALMODState:
    """
    Update the VALMOD state with information from a newly computed matrix profile.

    Reflects Algorithm 5 (UpdateVALMP) by recording distances, indices, and
    augmenting the motif dictionary with the top-p candidates at the current
    length. Motif keys are canonicalized to avoid duplicates from symmetric
    matches.

    Parameters
    ----------
    state : VALMODState
        Mutable state carrying accumulated VALMP information.
    length : int
        Subsequence length associated with the provided profile.
    matrix_profile : np.ndarray
        Distance values for each subsequence start index.
    profile_indices : np.ndarray
        Index of the best match for each subsequence.
    p : int
        Number of motifs to retain from this profile.

    Returns
    -------
    VALMODState
        Updated state with motif and profile caches refreshed.
    """
    state.distances[length] = matrix_profile
    state.indices[length] = profile_indices
    if length not in state.lengths:
        state.lengths.append(length)

    valid = np.where(profile_indices >= 0)[0]
    if valid.size == 0:
        return state

    order = valid[np.argsort(matrix_profile[valid])]
    top_order = order[: min(p, order.size)]
    for idx in top_order:
        match_idx = int(profile_indices[idx])
        motif_key = (length, min(idx, match_idx), max(idx, match_idx))
        motif_distance = float(matrix_profile[idx])
        stored = state.motifs.get(motif_key)
        if stored is None or motif_distance < stored["distance"]:
            state.motifs[motif_key] = {
                "length": length,
                "start_idx": int(idx),
                "match_idx": match_idx,
                "distance": motif_distance,
            }
    return state


def _initialize_state() -> VALMODState:
    """
    Create an empty VALMODState with deterministic containers.
    """
    return VALMODState(distances={}, indices={}, motifs={}, lengths=[])


def valmod(
    series_ids: Sequence[int],
    levels: np.ndarray,
    weights: np.ndarray,
    l_min: int,
    l_max: int,
    p: int,
    n_dp: Optional[int] = None,
    exclusion_fraction: float = 0.5,
) -> Dict[str, object]:
    """
    Run the VALMOD orchestration loop (Algorithm 1) with UIHE distances.

    Parameters
    ----------
    series_ids : Sequence[int]
        Encoded activity identifiers as in `util.ui_stump.stump` (Algorithm 1).
    levels : np.ndarray
        Lookup table that maps identifiers to hierarchy level codes.
    weights : np.ndarray
        Hierarchy-aware weights aligned with the columns of `levels`.
    l_min, l_max : int
        Minimum and maximum subsequence lengths to evaluate.
    p : int
        Number of motifs retained per length and for the final VALMP summary.
    n_dp : int, optional
        Count of cached distance profiles (defaults to `p` when None).
    exclusion_fraction : float, optional
        Self-match exclusion window expressed as a fraction of length.

    Returns
    -------
    dict
        Dictionary with matrix profile summaries:

        - ``distances`` : dict[int, np.ndarray]
        - ``indices``   : dict[int, np.ndarray]
        - ``lengths``   : list[int]
        - ``motifs``    : list[dict[str, float]]
    """
    if l_min < 2:
        raise ValueError("l_min must be at least 2.")
    if l_min > l_max:
        raise ValueError("l_min cannot exceed l_max.")
    if p < 1:
        raise ValueError("p must be positive.")
    if n_dp is None:
        n_dp = p

    state = _initialize_state()
    activity_matrix = _resolve_activity_matrix(series_ids, levels)
    listDP: Dict[int, np.ndarray] = {}

    for length in range(l_min, l_max + 1):
        mp, mpi, listDP, _ = compute_sub_matrix_profile(
            activity_matrix=activity_matrix,
            n_dp=n_dp,
            listDP=listDP,
            new_length=length,
            p=p,
            weights=weights,
            exclusion_fraction=exclusion_fraction,
        )
        state = update_valmp(
            state=state,
            length=length,
            matrix_profile=mp,
            profile_indices=mpi,
            p=p,
        )

    motifs_sorted = sorted(state.motifs.values(), key=lambda item: item["distance"])
    motifs_trimmed = motifs_sorted[: min(p, len(motifs_sorted))]

    result = {
        "distances": state.distances,
        "indices": state.indices,
        "lengths": sorted(state.lengths),
        "motifs": motifs_trimmed,
    }
    return result


if __name__ == "__main__":
    import numpy as _np

    # Example: simulated UI hierarchy ids/levels mirroring ui_stump.stump inputs.
    rng = _np.random.default_rng(seed=42)
    vocab_size = 50
    n_levels = 3
    levels = rng.integers(low=0, high=6, size=(vocab_size, n_levels))
    series_ids = rng.integers(low=0, high=vocab_size, size=100)
    weights = _np.array([4.0, 2.0, 1.0], dtype=_np.float64)
    result = valmod(series_ids, levels, weights, l_min=5, l_max=10, p=3)
    print("Top motifs:", result["motifs"][:3])
