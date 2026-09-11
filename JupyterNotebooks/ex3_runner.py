"""
EX3 runner: LoCoMotif vs. brute-force MP vs. Pan Matrix Profile on Leno UI logs.

All design decisions and their paper-facing justification live in
``EX3_design_decisions.md`` next to this file. Keep that file in sync.

Each ``run_*`` function receives already-encoded inputs and returns
``(discovered_segments, discovery_seconds)``. The driver
``run_all_for_log`` handles loading, encoding, PCA, evaluation, and
combines everything into one DataFrame row per (log, algorithm).
"""

import os
import sys
import time
import math
import warnings

import numpy as np
import pandas as pd

# Make sibling utilities importable when this module is imported from the
# notebook (which sits in JupyterNotebooks/).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import stumpy
from sklearn.decomposition import PCA

import locomotif.locomotif as locomotif

from util import valmod_uihe as valmod_util
from util import GrammarBasedUtil as grammar_util

warnings.filterwarnings("ignore", category=UserWarning)

LENO_DIR = os.path.join(_REPO_ROOT, "logs", "leno")

# Hierarchy columns for the Leno SmartRPA-format CSVs in logs/leno/.
# Mirrors the isActionLogger branch of util.util.read_data_for_processing.
_LENO_HIERARCHY = [
    ["targetApp"],
    ["url", "target.workbookName"],
    ["target.sheetName"],
    ["target.id", "target.tagName", "target.type", "target.name", "target.href"],
    ["eventType"],
]


def load_leno_log(log_stem: str):
    """Load a Leno log + its ground truth by file stem (without ``.csv``)."""
    log_path = os.path.join(LENO_DIR, f"{log_stem}.csv")
    gt_path = os.path.join(LENO_DIR, f"{log_stem}_ground_truth.csv")

    file_df = pd.read_csv(log_path, sep=";")
    gt = pd.read_csv(gt_path, sep=";")
    gt["start_index"] = gt["start_index"].astype(int)
    gt["length"] = gt["length"].astype(int)
    gt["end_index"] = gt["start_index"] + gt["length"] - 1

    hierarchy_columns = [c for sub in _LENO_HIERARCHY for c in sub if c in file_df.columns]
    log_df = grammar_util.symbolize_UILog(file_df, hierarchy_columns)
    return log_df, gt, hierarchy_columns


def encode_w2v(log_df: pd.DataFrame, hierarchy_columns: list) -> np.ndarray:
    """Reuse the same Word2Vec encoder ``experiment.py`` uses (encoding_method=1)."""
    active_cols = [c for c in hierarchy_columns if log_df[c].nunique() != 0]
    tokens = sum(log_df[c].nunique() for c in active_cols)
    vector_size = max(2, round(math.sqrt(tokens)))
    encoded = valmod_util.encode_word2vec(
        log_df,
        orderedColumnsList=active_cols,
        vector_size=vector_size,
        completeCorpusLog=log_df,
    )
    X = encoded.filter(like="w2v_").to_numpy().astype(float)
    return X, vector_size


def zscore(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=1)
    sd = np.where(sd == 0, 1.0, sd)
    return (X - mu) / sd


def pca_1d(X_z: np.ndarray):
    pca = PCA(n_components=1)
    ts_1d = pca.fit_transform(X_z).ravel().astype(float)
    ev = float(pca.explained_variance_ratio_[0])
    # PMP/MP also z-norm internally, but z-norming the full 1D series first
    # prevents PCA scale drift from biasing distances.
    if ts_1d.std(ddof=1) > 0:
        ts_1d = (ts_1d - ts_1d.mean()) / ts_1d.std(ddof=1)
    return ts_1d, ev


# ---------------------------------------------------------------------------
# Aggregation helper: top-k by lowest distance with IoU>0.5 NMS
# ---------------------------------------------------------------------------

def _iou(a, b) -> float:
    s = max(a[0], b[0])
    e = min(a[1], b[1])
    inter = max(0, e - s + 1)
    if inter == 0:
        return 0.0
    union = (a[1] - a[0] + 1) + (b[1] - b[0] + 1) - inter
    return inter / union


def _nms_topk(candidates, k: int, iou_thresh: float = 0.5):
    """``candidates``: list of ``(distance, (start, end))``. Sort ascending by
    distance, accept if IoU with every already-accepted < ``iou_thresh``."""
    accepted = []
    for _, seg in sorted(candidates, key=lambda x: x[0]):
        if all(_iou(seg, prev) < iou_thresh for prev in accepted):
            accepted.append(seg)
            if len(accepted) >= k:
                break
    return accepted


# ---------------------------------------------------------------------------
# LoCoMotif: multivariate, no PCA
# ---------------------------------------------------------------------------

def run_locomotif(X_z: np.ndarray, l_min: int, l_max: int, k: int, rho: float = 0.7):
    t0 = time.perf_counter()
    motif_sets = locomotif.apply_locomotif(X_z, l_min=l_min, l_max=l_max, rho=rho)
    elapsed = time.perf_counter() - t0

    # Flatten in returned order (LoCoMotif's own ranking). Each motif_set is
    # ((rep_start, rep_end), [(start, end), ...]); take occurrences in order.
    flat = []
    for _rep, occurrences in motif_sets:
        for s, e in occurrences:
            flat.append((int(s), int(e)))
    # Deduplicate while preserving order; trim to k.
    seen = set()
    discovered = []
    for seg in flat:
        if seg in seen:
            continue
        seen.add(seg)
        discovered.append(seg)
        if len(discovered) >= k:
            break
    return discovered, elapsed


# ---------------------------------------------------------------------------
# Brute-force MP: stumpy.stump at every m in [l_min, l_max]
# ---------------------------------------------------------------------------

def run_brute_mp(ts_1d: np.ndarray, l_min: int, l_max: int, k: int,
                 max_matches_per_m: int = 10):
    t0 = time.perf_counter()
    candidates = []  # (distance, (start, end))
    for m in range(int(l_min), int(l_max) + 1):
        if m < 3 or m > len(ts_1d) // 2:
            continue
        mp = stumpy.stump(ts_1d, m=m, normalize=True)
        P = mp[:, 0].astype(float)
        # Some entries can be inf if the subsequence is constant.
        if not np.isfinite(P).any():
            continue
        # cutoff=np.inf and max_distance=np.inf: do NOT prune candidates
        # statistically; the oracle-k + NMS step (per design §7) does the
        # selection. Without this override stumpy returns very few motifs
        # when the MP distribution is wide, defeating the oracle-k policy.
        d_motifs, i_motifs = stumpy.motifs(
            ts_1d, P, max_motifs=k, max_matches=max_matches_per_m,
            cutoff=np.inf, max_distance=np.inf, normalize=True,
        )
        for row_d, row_i in zip(d_motifs, i_motifs):
            for dist, start in zip(row_d, row_i):
                if start < 0 or not np.isfinite(dist):
                    continue
                candidates.append((float(dist), (int(start), int(start + m - 1))))
    discovered = _nms_topk(candidates, k=k, iou_thresh=0.5)
    elapsed = time.perf_counter() - t0
    return discovered, elapsed


# ---------------------------------------------------------------------------
# Multivariate Matrix Profile: stumpy.mstump on the full W2V matrix,
# swept over m. Same input contract as LoCoMotif — controls for PCA(1)
# dimensionality starvation that handicaps run_brute_mp.
# ---------------------------------------------------------------------------

def _extract_motifs_from_profile(mp_row: np.ndarray, m: int, n_pairs: int = 10):
    """Greedy top-n extraction from a 1D matrix profile: argmin → mask an
    exclusion zone of width m//4 around the pick → repeat."""
    P = np.asarray(mp_row, dtype=float).copy()
    excl = max(1, m // 4)
    results = []
    for _ in range(n_pairs):
        if not np.isfinite(P).any():
            break
        idx = int(np.nanargmin(P))
        dist = float(P[idx])
        if not np.isfinite(dist):
            break
        results.append((dist, (idx, idx + m - 1)))
        lo = max(0, idx - excl)
        hi = min(len(P), idx + excl + 1)
        P[lo:hi] = np.inf
    return results


def run_mstamp(X_z: np.ndarray, l_min: int, l_max: int, k: int,
               max_matches_per_m: int = 10):
    t0 = time.perf_counter()
    # stumpy.mstump expects (d, n); X_z is (n, d).
    T = np.ascontiguousarray(X_z.T)
    n_len = T.shape[1]
    candidates = []
    for m in range(int(l_min), int(l_max) + 1):
        if m < 3 or m > n_len // 2:
            continue
        mps, _ = stumpy.mstump(T, m=m, normalize=True)
        # Use the full-subspace (d-1) row — distance computed using all dims,
        # matching the input that LoCoMotif consumes.
        mp_row = np.asarray(mps[-1], dtype=float)
        if not np.isfinite(mp_row).any():
            continue
        for dist, seg in _extract_motifs_from_profile(mp_row, m, n_pairs=max_matches_per_m):
            candidates.append((dist, seg))
    discovered = _nms_topk(candidates, k=k, iou_thresh=0.5)
    elapsed = time.perf_counter() - t0
    return discovered, elapsed


# ---------------------------------------------------------------------------
# Pan Matrix Profile: stumpy.stimp + pooling across every m in the pan stack
# ---------------------------------------------------------------------------

def _modal_m(discovered) -> int | None:
    if not discovered:
        return None
    lengths = [seg[1] - seg[0] + 1 for seg in discovered]
    return int(max(set(lengths), key=lengths.count))


def run_pmp(ts_1d: np.ndarray, l_min: int, l_max: int, k: int, percentage: float,
            max_matches: int = 10):
    """Build the pan matrix profile, extract motifs from *every* row, pool across
    m, NMS by IoU > 0.5. This implements the §8 design-doc claim. The earlier
    version that collapsed to a single best-m and re-ran ``stumpy.stump`` is
    why ``PMP_full`` and ``PMP_sampled`` produced identical numbers."""
    t0 = time.perf_counter()
    pmp = stumpy.stimp(
        ts_1d,
        min_m=int(l_min),
        max_m=int(l_max),
        step=1,
        percentage=percentage,
        pre_scrump=True,
        normalize=True,
    )
    n_m = int(l_max) - int(l_min) + 1
    for _ in range(n_m):
        pmp.update()

    candidates = []
    for row_idx, m in enumerate(pmp.M_):
        m_int = int(m)
        valid_len = len(ts_1d) - m_int + 1
        if valid_len <= 0:
            continue
        P_full = np.asarray(pmp.PAN_[row_idx], dtype=float)
        # PAN_ rows are padded to a uniform width; trim to the matrix profile's
        # actual length n - m + 1 before reading.
        P = P_full[:valid_len]
        if not np.isfinite(P).any():
            continue
        d_motifs, i_motifs = stumpy.motifs(
            ts_1d, P, max_motifs=k, max_matches=max_matches,
            cutoff=np.inf, max_distance=np.inf, normalize=True,
        )
        for row_d, row_i in zip(d_motifs, i_motifs):
            for dist, start in zip(row_d, row_i):
                if start < 0 or not np.isfinite(dist):
                    continue
                candidates.append((float(dist), (int(start), int(start + m_int - 1))))

    discovered = _nms_topk(candidates, k=k, iou_thresh=0.5)
    elapsed = time.perf_counter() - t0
    return discovered, elapsed, _modal_m(discovered)


# ---------------------------------------------------------------------------
# Random baseline: chance-level reference for the F1 scale
# ---------------------------------------------------------------------------

def run_random(T_len: int, l_min: int, l_max: int, k: int, gt: pd.DataFrame,
               n_seeds: int = 100, base_seed: int = 42):
    """Sample ``k`` random segments per seed, evaluate against ground truth,
    return mean and std across seeds. Length per segment is Uniform[l_min,
    l_max]; start is Uniform[0, T - length]. Same oracle ``k`` and length
    range as every other algorithm — answers "is your F1 better than guessing
    under the same budget?"."""
    t0 = time.perf_counter()
    f1s, ps, rs, mean_ious, tps, fps, fns = [], [], [], [], [], [], []
    for s in range(n_seeds):
        rng = np.random.default_rng(base_seed + s)
        lengths = rng.integers(int(l_min), int(l_max) + 1, size=k)
        max_starts = T_len - lengths
        # Reject seeds where lengths exceed T (defensive — shouldn't happen for these logs).
        if (max_starts < 0).any():
            continue
        starts = np.array([int(rng.integers(0, int(ms) + 1)) for ms in max_starts])
        segments = [(int(st), int(st + ln - 1)) for st, ln in zip(starts, lengths)]
        m = evaluate(segments, gt)
        f1s.append(m["f1"]); ps.append(m["precision"]); rs.append(m["recall"])
        mean_ious.append(m["mean_iou_tp"])
        tps.append(m["tp"]); fps.append(m["fp"]); fns.append(m["fn"])
    elapsed = time.perf_counter() - t0
    return {
        "tp": float(np.mean(tps)), "fp": float(np.mean(fps)), "fn": float(np.mean(fns)),
        "precision": float(np.mean(ps)), "recall": float(np.mean(rs)),
        "f1": float(np.mean(f1s)),
        "mean_iou_tp": float(np.mean(mean_ious)),
        "n_discovered": k,
        "f1_std": float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0,
        "precision_std": float(np.std(ps, ddof=1)) if len(ps) > 1 else 0.0,
        "recall_std": float(np.std(rs, ddof=1)) if len(rs) > 1 else 0.0,
        "n_seeds": len(f1s),
    }, elapsed


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(discovered, ground_truth: pd.DataFrame, iou_threshold: float = 0.8) -> dict:
    if not discovered:
        return {
            "tp": 0, "fp": 0, "fn": int(len(ground_truth)),
            "precision": 0.0, "recall": 0.0, "f1": 0.0, "mean_iou_tp": 0.0,
            "n_discovered": 0,
        }
    motif_ranges = [[s, e] for s, e in discovered]
    res = grammar_util.evaluate_motifs(
        motif_ranges, ground_truth,
        overlap_threshold=iou_threshold, overlap_type="iou",
    )
    tp, fp, fn = int(res["tp"]), int(res["fp"]), int(res["fn"])
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    table = res["overlap_table"]
    if "is_best_match" in table.columns:
        matched = table[table["is_best_match"]]
        mean_iou_tp = float(matched["iou"].mean()) if len(matched) else 0.0
    else:
        mean_iou_tp = 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "mean_iou_tp": mean_iou_tp,
        "n_discovered": len(discovered),
    }


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def run_all_for_log(log_stem: str, verbose: bool = True) -> pd.DataFrame:
    log_df, gt, hierarchy_columns = load_leno_log(log_stem)

    t_pre = time.perf_counter()
    X, w2v_dim = encode_w2v(log_df, hierarchy_columns)
    X_z = zscore(X)
    ts_1d, pca_ev = pca_1d(X_z)
    pre_elapsed = time.perf_counter() - t_pre

    l_min = max(3, int(gt["length"].min()))
    l_max = int(gt["length"].max())
    k = int(len(gt))

    if verbose:
        print(f"[{log_stem}] T={len(log_df)} d={w2v_dim} l_min={l_min} l_max={l_max} k={k} pca_ev={pca_ev:.3f}")

    rows = []

    # --- LoCoMotif ---
    try:
        seg, t_disc = run_locomotif(X_z, l_min, l_max, k)
        metrics = evaluate(seg, gt)
        rows.append({"algorithm": "LoCoMotif", "discovery_time_s": t_disc, "best_m": None, **metrics})
    except Exception as exc:
        rows.append({"algorithm": "LoCoMotif", "error": str(exc),
                     "discovery_time_s": np.nan, "best_m": None,
                     "tp": np.nan, "fp": np.nan, "fn": np.nan,
                     "precision": np.nan, "recall": np.nan, "f1": np.nan,
                     "mean_iou_tp": np.nan, "n_discovered": 0})

    # --- Brute-force MP (PCA(1)) ---
    try:
        seg, t_disc = run_brute_mp(ts_1d, l_min, l_max, k)
        metrics = evaluate(seg, gt)
        rows.append({"algorithm": "BruteMP", "discovery_time_s": t_disc, "best_m": None, **metrics})
    except Exception as exc:
        rows.append({"algorithm": "BruteMP", "error": str(exc),
                     "discovery_time_s": np.nan, "best_m": None,
                     "tp": np.nan, "fp": np.nan, "fn": np.nan,
                     "precision": np.nan, "recall": np.nan, "f1": np.nan,
                     "mean_iou_tp": np.nan, "n_discovered": 0})

    # --- Multivariate MP (mSTAMP, same input as LoCoMotif) ---
    try:
        seg, t_disc = run_mstamp(X_z, l_min, l_max, k)
        metrics = evaluate(seg, gt)
        rows.append({"algorithm": "MStampBrute", "discovery_time_s": t_disc, "best_m": None, **metrics})
    except Exception as exc:
        rows.append({"algorithm": "MStampBrute", "error": str(exc),
                     "discovery_time_s": np.nan, "best_m": None,
                     "tp": np.nan, "fp": np.nan, "fn": np.nan,
                     "precision": np.nan, "recall": np.nan, "f1": np.nan,
                     "mean_iou_tp": np.nan, "n_discovered": 0})

    # --- PMP, full SKIMP (percentage=1.0) ---
    try:
        seg, t_disc, best_m = run_pmp(ts_1d, l_min, l_max, k, percentage=1.0)
        metrics = evaluate(seg, gt)
        rows.append({"algorithm": "PMP_full", "discovery_time_s": t_disc, "best_m": best_m, **metrics})
    except Exception as exc:
        rows.append({"algorithm": "PMP_full", "error": str(exc),
                     "discovery_time_s": np.nan, "best_m": None,
                     "tp": np.nan, "fp": np.nan, "fn": np.nan,
                     "precision": np.nan, "recall": np.nan, "f1": np.nan,
                     "mean_iou_tp": np.nan, "n_discovered": 0})

    # --- PMP, sampled SKIMP (percentage=0.01) ---
    try:
        seg, t_disc, best_m = run_pmp(ts_1d, l_min, l_max, k, percentage=0.01)
        metrics = evaluate(seg, gt)
        rows.append({"algorithm": "PMP_sampled", "discovery_time_s": t_disc, "best_m": best_m, **metrics})
    except Exception as exc:
        rows.append({"algorithm": "PMP_sampled", "error": str(exc),
                     "discovery_time_s": np.nan, "best_m": None,
                     "tp": np.nan, "fp": np.nan, "fn": np.nan,
                     "precision": np.nan, "recall": np.nan, "f1": np.nan,
                     "mean_iou_tp": np.nan, "n_discovered": 0})

    # --- Random baseline ---
    try:
        metrics, t_disc = run_random(len(log_df), l_min, l_max, k, gt, n_seeds=100)
        rows.append({"algorithm": "Random", "discovery_time_s": t_disc, "best_m": None, **metrics})
    except Exception as exc:
        rows.append({"algorithm": "Random", "error": str(exc),
                     "discovery_time_s": np.nan, "best_m": None,
                     "tp": np.nan, "fp": np.nan, "fn": np.nan,
                     "precision": np.nan, "recall": np.nan, "f1": np.nan,
                     "mean_iou_tp": np.nan, "n_discovered": 0})

    base = {
        "log_name": log_stem,
        "log_length": int(len(log_df)),
        "n_gt_motifs": k,
        "l_min": l_min,
        "l_max": l_max,
        "w2v_dim": w2v_dim,
        "pca_explained_variance": pca_ev,
        "preprocessing_time_s": pre_elapsed,
    }
    df = pd.DataFrame(rows)
    for col, val in base.items():
        df[col] = val
    front = ["log_name", "log_length", "n_gt_motifs", "algorithm",
             "l_min", "l_max", "w2v_dim", "pca_explained_variance",
             "preprocessing_time_s", "discovery_time_s", "best_m",
             "tp", "fp", "fn", "precision", "recall", "f1",
             "mean_iou_tp", "n_discovered"]
    cols = front + [c for c in df.columns if c not in front]
    return df[cols]


LENO_LOGS = [
    "202511_SR_RT_plus",
    "202511_SR_RT_plus_extended",
    "202511_SR_RT_parallel",
    "202511_SR_RT_parallel_extended",
]


def run_all_logs(logs=None, verbose: bool = True) -> pd.DataFrame:
    if logs is None:
        logs = LENO_LOGS
    frames = []
    for stem in logs:
        if verbose:
            print(f"\n=== {stem} ===")
        frames.append(run_all_for_log(stem, verbose=verbose))
    return pd.concat(frames, ignore_index=True)
