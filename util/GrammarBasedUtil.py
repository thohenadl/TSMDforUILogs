import pandas as pd
from collections import Counter,defaultdict
import numpy as np
import itertools
import matplotlib.pyplot as plt


from collections import defaultdict   # <-- needed for defaultdict
from typing import List, Dict, Any

##########################
### Grammer Viz Code #####
##########################

# Function to convert numeric code to Excel-style letters (A, B, ..., Z, AA, AB, ...)
def num_to_letters(n, lowercase=False):
        s = ""
        while True:
            n, r = divmod(n, 26)
            if lowercase:
                s = chr(97 + r) + s
            else:
                s = chr(65 + r) + s
            if n == 0:
                break
            n -= 1
        return s

def symbolize_UILog(log: pd.DataFrame, hierarchy_cols: list) -> pd.DataFrame:
    # Combine all hierarchy columns into a single string key (vectorized)
    combo_series = log[hierarchy_cols].astype(str).agg('|'.join, axis=1)
    # Compute categorical codes
    codes = pd.Categorical(combo_series).codes
    # Precompute all unique letter codes
    unique_codes = [num_to_letters(i) for i in range(len(set(codes)))]
    # Map numeric codes to letters efficiently
    log["symbol"] = [unique_codes[i] for i in codes]
    # Add a 'count' column showing frequency of each unique combination
    log["count"] = (
        log
        .groupby(hierarchy_cols, dropna=False)
        [hierarchy_cols[0]]
        .transform("count")
    )
    return log

def build_2gram_dataframe(log, column="symbol"):
    """
    Build a dataframe of all 2-grams with their counts and index positions.
    Each row includes:
        2-gram : tuple of (symbol_i, symbol_i+1)
        count  : frequency
        indexes: list of (i, i+1) index tuples where the 2-gram occurs
    """
    symbols = log[column].astype(str).tolist()
    two_grams = list(zip(symbols, symbols[1:]))

    # Count frequencies
    two_gram_counts = Counter(two_grams)

    # Record positions for each 2-gram
    two_gram_positions = defaultdict(list)
    for i, pair in enumerate(two_grams):
        two_gram_positions[pair].append((i, i + 1))

    # Construct dataframe
    two_gram_df = (
        pd.DataFrame([
            {"2-gram": pair, "count": count, "indexes": two_gram_positions[pair]}
            for pair, count in two_gram_counts.items()
        ])
        .sort_values(by="count", ascending=False)
        .reset_index(drop=True)
    )

    return two_gram_df

def re_pair(log: pd.DataFrame, printing: bool = False):
    """
    Perform Re-Pair compression on the 'symbol' column of the log DataFrame.
    Returns:
        encoding_df: DataFrame with columns ['pair', 'new_symbol', 'count']
        encoded_symbols: List of symbols after Re-Pair compression
        two_2gram_df: DataFrame of 2-grams"""
    
    encoding_df = pd.DataFrame(columns=["pair", "new_symbol", "count"])
    used_lowercase = set()

    symbols = log["symbol"].astype(str).tolist()
    # Create 2-grams (adjacent symbol pairs)
    two_gram_df = build_2gram_dataframe(log, column="symbol")
    j = 0
    while any(two_gram_df["count"] >= 2):
        j += 1
        for index, row in two_gram_df.iterrows():
            
            if row["count"] >= 2:
                lowercase_letter = num_to_letters(len(used_lowercase), lowercase=True)
                used_lowercase.add(lowercase_letter)
                entry = {
                    "pair": row["2-gram"],
                    "new_symbol": lowercase_letter,
                    "count": row["count"]
                }
                encoding_df = pd.concat([encoding_df, pd.DataFrame([entry])], ignore_index=True)
                for i in range(len(symbols) - 1):
                    if (symbols[i], symbols[i + 1]) == entry["pair"]:
                        symbols[i] = lowercase_letter # Replace first symbol with new symbol
                        symbols[i + 1] = None # Remove second symbol
        if printing:
            print(len(encoding_df), " unique pairs found so far. Rule-Level: ", j)
        symbols = [s for s in symbols if s is not None] # Remove None entries
        two_grams = list(zip(symbols, symbols[1:]))
        two_gram_counts = Counter(two_grams)

        two_gram_df = (
            pd.DataFrame(two_gram_counts.items(), columns=["2-gram", "count"])
            .sort_values(by="count", ascending=False)
            .reset_index(drop=True)
        )

    return encoding_df, symbols, two_gram_df

def re_pair_decode_symbol(symbol, encoding_df, printing=False, depth=0):
    """
    Recursively decodes a single encoded symbol using the encoding_df returned by re_pair().
    Prints the decoding path to show how the symbol expands through the hierarchy.
    Returns a list of the original base symbols.
    """
    # Build decoding dictionary
    decode_map = {
        row["new_symbol"]: tuple(row["pair"])
        for _, row in encoding_df.iterrows()
    }

    def _decode(sym, level):
        indent = "  " * level
        if sym not in decode_map:
            if printing:
                print(f"{indent}{sym} → [base symbol]")
            return [sym]

        left, right = decode_map[sym]
        if printing:
            print(f"{indent}{sym} → ({left}, {right})")
        left_decoded = _decode(left, level + 1)
        right_decoded = _decode(right, level + 1)
        return left_decoded + right_decoded

    
    decoded = _decode(symbol, depth)
    if printing:
        print(f"\nDecoding path for symbol '{symbol}':")
        print(f"Final expansion: {decoded}\n")
    return decoded

def re_pair_decode_all(encoding_df):
    """
    Decode all grammar rules exactly once (consistent with re_pair_decode_symbol).
    Returns {new_symbol: list of base symbols}.
    """
    # Ensure uniqueness and order
    encoding_df = encoding_df.drop_duplicates(subset="new_symbol", keep="last")

    decode_map = {
        row["new_symbol"]: tuple(row["pair"]) for _, row in encoding_df.iterrows()
    }

    cache = {}

    def _decode(sym):
        if sym in cache:
            return cache[sym]
        if sym not in decode_map:
            cache[sym] = (sym,)
            return cache[sym]
        left, right = decode_map[sym]
        result = _decode(left) + _decode(right)
        cache[sym] = result
        return result

    for sym in decode_map:
        _decode(sym)

    # Return lists for compatibility
    return {k: list(v) for k, v in cache.items()}


# ---- Evaluation Methods ----

def generate_density_count(encoding_df, log, column_name="rule_density_count"):
    log[column_name] = 0
    decoded_rules = re_pair_decode_all(encoding_df) 
    for sym, decoding in decoded_rules.items():
        # Sliding window search for the sequence
        seq_len = len(decoding)
        symbols = log["symbol"].astype(str).tolist()
        for i in range(len(symbols) - seq_len + 1):
            if symbols[i:i + seq_len] == decoding:
                log.loc[i:i + seq_len - 1, column_name] += 1
    return log

def find_rules_with_symbol(encoding_df, symbol, recursive=False):
    """
    Find all grammar rules in encoding_df that include a given symbol.

    Parameters
    ----------
    encoding_df : pd.DataFrame
        Must contain columns ['pair', 'new_symbol'] where 'pair' is a tuple (left, right).
    symbol : str
        The symbol to search for (e.g. 'A', 'a', 'f', etc.).
    recursive : bool, default False
        If True, also returns all rules that depend (directly or indirectly) on the symbol.

    Returns
    -------
    related_rules : pd.DataFrame
        Subset of encoding_df containing all matching rules.
           
              
    Example Usage
    -------
    find_rules_with_symbol(encoding_df, "a", recursive=True)   
    """

    # direct matches: any rule whose left or right contains the symbol
    direct = encoding_df[
        encoding_df["pair"].apply(lambda p: symbol in p)
    ].copy()

    if not recursive:
        return direct

    # recursive search: find all rules whose pairs contain the symbol, then all rules that
    # contain those new_symbols, and so on
    seen = set()
    to_check = {symbol}

    while to_check:
        current = to_check.pop()
        new_matches = encoding_df[
            encoding_df["pair"].apply(lambda p: current in p)
        ]
        for _, row in new_matches.iterrows():
            new_sym = row["new_symbol"]
            if new_sym not in seen:
                seen.add(new_sym)
                to_check.add(new_sym)

    # all rules whose new_symbol is in seen
    recursive_rules = encoding_df[
        encoding_df["new_symbol"].isin(seen)
    ].copy()

    return recursive_rules


def find_max_density_groups(
    log, 
    column="rule_density_count",
    method="threshold",
    relative_threshold=None, 
    absolute_threshold=None,
    merge_gap=None,
    percentile_threshold: float = 0.95,
):
    """
    Find indices of maximum (or near-maximum) rule density and group consecutive ones.

    Parameters
    ----------
    log : pd.DataFrame
        DataFrame containing the rule density column.
    column : str, default="rule_density_count"
        Name of the column with rule density values.
    method : str, default="threshold"
        Method to determine maximum density groups. Options are "threshold", "percentile", or "MAD".
    relative_threshold : float, optional
        Include all values >= (relative_threshold * max_density).
        For example, 0.9 means include all densities >= 90% of the max.
    absolute_threshold : float or int, optional
        Include all values >= (max_density - absolute_threshold).
        For example, 1 means include all densities within 1 count of the max.
    merge_gap : int, optional
        Maximum absolute gap between groups to merge them.
    percentile_threshold: float, default=0.95
        Percentile threshold for "percentile" method.
        
    Returns
    -------
    max_density : float
        The maximum density value.
    groups : list of lists
        Each inner list contains consecutive indices meeting the threshold.
    """

    # Maximum density value
    max_density = log[column].max()

    #Variant A: Threshold Based
    if method == "threshold":
        # Step 2: determine cutoff based on threshold
        if relative_threshold is not None:
            cutoff = max_density * relative_threshold
        elif absolute_threshold is not None:
            cutoff = max_density - absolute_threshold
        else:
            cutoff = max_density

    # Variant B: Percentile Based
    elif method == "percentile":
        cutoff = log[column].quantile(percentile_threshold)

    elif method == "MAD":
        # Step 2: determine cutoff
        # ---- MAD-based robust threshold ----
        vals = log[column]
        nz = vals[vals > 0]

        median_nz = nz.median()
        print("Median of non-zero densities:", median_nz)
        mad_nz = (nz - median_nz).abs().median()
        print("MAD of non-zero densities:", mad_nz)

        cutoff = median_nz + 3 * mad_nz
        print("Computed cutoff using median + 3*MAD:", cutoff)
    else:
        raise ValueError("Invalid method. Choose 'threshold', 'percentile', or 'MAD'.")


    # Step 3: get indices meeting the cutoff
    valid_indices = log.index[log[column] >= cutoff].tolist()
    if not valid_indices:
        return []
    # Step 4: group consecutive indices
    groups = []
    for _, group in itertools.groupby(
        enumerate(valid_indices), lambda x: x[0] - x[1]
    ):
        connected = [g[1] for g in group]
        groups.append(connected)

    if merge_gap is not None:
        # Merge groups if they are close (≤ merge_gap apart)
        merged = []
        for g in groups:
            if not merged:
                merged.append(g)
            else:
                if g[0] - merged[-1][-1] <= merge_gap:
                    merged[-1].extend(g)
                else:
                    merged.append(g)
        return max_density, merged
    
    return groups, None


def evaluate_motifs(max_groups, ground_truth, overlap_threshold=0.5):
    """
    Unified, efficient motif evaluation function.
    
    Parameters
    ----------
    max_groups : list of lists
        Each element is a list representing discovered motif indices (start..end).
        We assume: g[0] is start index, g[-1] is end index.

    ground_truth : pd.DataFrame
        Must contain columns ["start_index", "end_index"].

    overlap_threshold : float, optional
        Minimum overlap ratio to consider a motif and ground truth as matching.

    Returns
    -------
    dict with:
        - overlap_table (pd.DataFrame)
        - matched_pairs (list of (motif_id, gt_id))
        - tp, fp, fn (int)
        - intersection_ratio, intersection_abs
        - undercount_ratio, undercount_abs
        - over_detection_ratio, over_detection_abs
    """

    # ----------------------------------------------------------------------
    # 1) Extract motif and GT ranges
    # ----------------------------------------------------------------------
    motif_ranges = [(g[0], g[-1]) for g in max_groups]
    gt_ranges = list(zip(
        ground_truth["start_index"].tolist(),
        ground_truth["end_index"].tolist()
    ))

    M, G = len(motif_ranges), len(gt_ranges)

    motif_lengths = [e - s + 1 for s, e in motif_ranges]
    gt_lengths = [e - s + 1 for s, e in gt_ranges]

    # ----------------------------------------------------------------------
    # 2) Build Overlap Matrix (M x G)
    # ----------------------------------------------------------------------
    overlap = np.zeros((M, G), dtype=int)

    for i, (ms, me) in enumerate(motif_ranges):
        for j, (gs, ge) in enumerate(gt_ranges):
            overlap[i, j] = max(0, min(me, ge) - max(ms, gs) + 1)

    # ----------------------------------------------------------------------
    # 3) Build Tidy Overlap Table (for human inspection)
    # ----------------------------------------------------------------------
    rows = []
    for i, (ms, me) in enumerate(motif_ranges):
        mlen = motif_lengths[i]
        for j, (gs, ge) in enumerate(gt_ranges):
            glen = gt_lengths[j]
            ov = overlap[i, j]

            rows.append({
                "motif_id": i,
                "gt_id": j,
                "motif_start": ms,
                "motif_end": me,
                "gt_start": gs,
                "gt_end": ge,
                "motif_length": mlen,
                "gt_length": glen,
                "overlap": ov,
                "overlap_motif_ratio": ov / mlen if mlen > 0 else 0,
                "overlap_gt_ratio": ov / glen if glen > 0 else 0,
            })

    df = pd.DataFrame(rows)

    # ----------------------------------------------------------------------
    # 4) One-to-One Best Matching (Greedy by Overlap)
    # ----------------------------------------------------------------------
    candidate_pairs = df[df["overlap"] > overlap_threshold].sort_values("overlap", ascending=False)

    matched_motifs = set()
    matched_gts = set()
    matched_pairs = set()

    for _, row in candidate_pairs.iterrows():
        mi, gi = int(row["motif_id"]), int(row["gt_id"])
        if mi not in matched_motifs and gi not in matched_gts:
            matched_motifs.add(mi)
            matched_gts.add(gi)
            matched_pairs.add((mi, gi))

    df["is_best_match"] = df.apply(
        lambda r: (r["motif_id"], r["gt_id"]) in matched_pairs,
        axis=1
    )

    tp = len(matched_pairs)
    fp = M - tp
    fn = G - tp

    # ----------------------------------------------------------------------
    # 5) Metrics from Overlap Matrix
    # ----------------------------------------------------------------------
    # Motif-based (intersection and over-detection)
    intersection_abs = overlap.sum(axis=1)
    intersection_ratio = intersection_abs / np.maximum(motif_lengths, 1)

    overdet_abs = np.array(motif_lengths) - intersection_abs
    overdet_ratio = overdet_abs / np.maximum(motif_lengths, 1)

    # GT-based (undercount)
    under_abs = np.maximum(0, np.array(gt_lengths) - overlap.sum(axis=0))
    under_ratio = under_abs / np.maximum(gt_lengths, 1)

    # ----------------------------------------------------------------------
    # 6) Final Unified Result
    # ----------------------------------------------------------------------
    return {
        "overlap_table": df,
        "matched_pairs": list(matched_pairs),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "intersection_ratio": float(np.mean(intersection_ratio)) if M > 0 else 0,
        "intersection_abs": float(np.mean(intersection_abs)) if M > 0 else 0,
        "undercount_ratio": float(np.mean(under_ratio)) if G > 0 else 0,
        "undercount_abs": float(np.mean(under_abs)) if G > 0 else 0,
        "over_detection_ratio": float(np.mean(overdet_ratio)) if M > 0 else 0,
        "over_detection_abs": float(np.mean(overdet_abs)) if M > 0 else 0,
    }


def mark_overlaps_grammer_locomotif_indexed(df1: pd.DataFrame, df2: pd.DataFrame, col_df1: str = "original_df_range", col_df2: str = "group"):
    # 1) Build inverted index from df2
    index_map = defaultdict(set)   # number -> set of df2 row indices

    for j, r2 in enumerate(df2[col_df2]):
        for val in r2:
            index_map[val].add(j)

    # 2) For each df1 range, check if *any* value appears in index_map
    matches = []
    for r1 in df1[col_df1]:
        has_overlap = any(val in index_map for val in r1)
        matches.append(has_overlap)

    df1 = df1.copy()
    df1["grammer_motif_match"] = matches
    return df1

# Util in research: May be delated for publishing

def app_switch_miner(log: pd.DataFrame, rule_motifs: pd.DataFrame, comparison_cols: list):
    rule_motifs["lower_app_switch"] = -1
    rule_motifs["upper_app_switch"] = -1

    for i, row in rule_motifs.iterrows():
        start_index = row['start_index']
        rule_motifs.loc[i, 'lower_app_switch'] = int(start_index)
        # Go down until the app changes
        while start_index > 0 and log.loc[start_index, comparison_cols].equals(log.loc[start_index-1, comparison_cols]):
            rule_motifs.loc[i, 'lower_app_switch'] = int(start_index)
            start_index -= 1
        
        end_index = row['end_index']
        rule_motifs.loc[i, 'upper_app_switch'] = int(end_index)
        while end_index < len(log)-1 and log.loc[end_index, comparison_cols].equals(log.loc[end_index+1, comparison_cols]):
            rule_motifs.loc[i, 'upper_app_switch'] = int(end_index)
            end_index += 1

    return rule_motifs

def similar_path_up_down(
    df: pd.DataFrame,
    max_groups_df: pd.DataFrame,
    start_indices: List[int],
    end_indices: List[int],
    cols: List[str],
    min_pairs: int = 2,
) -> pd.DataFrame:
    """
    Compute, for each (start_index, end_index) pair, how many DOWN steps its
    start survives when ALL start indices are compared against each other,
    and how many UP steps its end survives when ALL end indices are compared
    against each other. At each step we keep only value-groups (by `cols`)
    with at least `min_pairs` members.
    """

    n = len(start_indices)
    assert n == len(end_indices), "start_indices and end_indices must have same length"

    # --- DOWN: operate only on the cohort of start indices (compare starts against starts) ---
    down_iters = defaultdict(int)  # pair_idx -> iterations survived
    active = [{"pair_idx": i, "cur": start_indices[i]} for i in range(n)]

    while True:
        # advance all active one step DOWN (previous rows)
        candidates = []
        for a in active:
            nxt = a["cur"] - 1
            if 0 <= nxt < len(df):
                candidates.append({"pair_idx": a["pair_idx"], "cur": nxt})

        if len(candidates) < min_pairs:
            break

        # group by column-value tuple
        groups = defaultdict(list)
        for c in candidates:
            row = df.loc[c["cur"], cols]
            key = tuple(row[col] for col in cols)
            groups[key].append(c)

        # keep only groups with >= min_pairs members
        new_active = []
        for members in groups.values():
            if len(members) >= min_pairs:
                new_active.extend(members)

        if len(new_active) < min_pairs:
            break

        # survivors increment their counters and continue
        for s in new_active:
            down_iters[s["pair_idx"]] += 1
        active = new_active

    # --- UP: operate only on the cohort of end indices (compare ends against ends) ---
    up_iters = defaultdict(int)  # pair_idx -> iterations survived
    active = [{"pair_idx": i, "cur": end_indices[i]} for i in range(n)]

    while True:
        # advance all active one step UP (following rows)
        candidates = []
        for a in active:
            nxt = a["cur"] + 1
            if 0 <= nxt < len(df):
                candidates.append({"pair_idx": a["pair_idx"], "cur": nxt})

        if len(candidates) < min_pairs:
            break

        # group by column-value tuple
        groups = defaultdict(list)
        for c in candidates:
            row = df.loc[c["cur"], cols]
            key = tuple(row[col] for col in cols)
            groups[key].append(c)

        # keep only groups with >= min_pairs members
        new_active = []
        for members in groups.values():
            if len(members) >= min_pairs:
                new_active.extend(members)

        if len(new_active) < min_pairs:
            break

        # survivors increment their counters and continue
        for s in new_active:
            up_iters[s["pair_idx"]] += 1
        active = new_active

    # assemble per-pair results
    max_groups_df["lower_pattern_switch"] = start_indices
    max_groups_df["upper_pattern_switch"] = end_indices
    for i in range(n):

        max_groups_df.loc[i, "lower_pattern_switch"] = start_indices[i] - down_iters.get(i, 0)
        max_groups_df.loc[i, "upper_pattern_switch"] = end_indices[i] + up_iters.get(i, 0)

    return max_groups_df

def join_discovery_with_ground_truth(final_discovery_result, ground_truth, motif_df):
    """
    Join the final discovery result's overlap table with ground truth and motif details.
    Parameters
    ----------
    final_discovery_result : pd.DataFrame
        Output of evaluate_motifs(), containing "overlap_table".
    ground_truth : pd.DataFrame
        Ground truth DataFrame without "gt_id" column.
    motif_df : pd.DataFrame
        DataFrame of discovered motifs with index as "motif_id".
    """
    overlap_table_locomotif = final_discovery_result["overlap_table"]

    gt_result_join_overlap = overlap_table_locomotif[overlap_table_locomotif["is_best_match"] == True]
    try: 
        ground_truth.reset_index(inplace=True)
    except:
        pass
    ground_truth["gt_id"] = ground_truth.index
    gt_result_join_overlap = gt_result_join_overlap.merge(
        ground_truth,
        how="outer",
        left_on="gt_id",
        right_on="gt_id",   # must exist in ground_truth
        suffixes=("", "_gt")
    )

    gt_result_join_overlap = gt_result_join_overlap.merge(
        motif_df,                 # use motif_df with 0..M-1 index
        how="outer",
        left_on="motif_id",
        right_index=True,
        suffixes=("", "_locomotif")
    )
    return gt_result_join_overlap

def cluster_level_metrics(df):
    """
    Calculate precision, recall, and F1-score at the cluster level from LOCOmotif.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns:
            - 'cluster_id': identifier for each cluster
            - 'caseid': identifier for each case (can be NaN)
            - 'total_occurances': total required occurrences for each caseid (can be NaN)
    """
    results = []

    for clust, g in df.groupby('cluster_id'):
        if pd.isna(clust):
            continue

        # Caseids that appear in this particular cluster
        caseids_in_cluster = g['caseid'].dropna().unique()

        if len(caseids_in_cluster) == 0:
            required_total = 0
        else:
            # Sum the required total occurrences for those caseids (from whole df)
            required_total = (
                df.loc[df['caseid'].isin(caseids_in_cluster), 'total_occurances']
                .dropna()
                .max()
            )

        # True positives: rows where this cluster has a caseid
        tp = g['caseid'].notna().sum()

        # False negatives: required_total - TP (lower bounded by 0)
        fn = max(required_total - tp, 0)

        # False positives: cluster predicted but caseid missing
        fp = g['caseid'].isna().sum()

        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

        results.append([clust, precision, recall, f1, tp, fp, fn, required_total])

    return pd.DataFrame(
        results,
        columns=["cluster_id", "precision", "recall", "f1", "tp", "fp", "fn", "required_total"]
    )

def motif_level_metrics(df):
    """
    Calculate precision, recall, and F1-score at the motif level.
    
    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns:
            - 'motif': identifier for each motif
            - 'motif_id': identifier for each motif instance (can be NaN)
            - 'caseid': identifier for each case (can be NaN)
            - 'cluster_id': identifier for each cluster (can be NaN)
            - 'total_occurances': total required occurrences for each caseid (can be NaN)
    """
    results = []

    for motif, g in df.groupby("motif"):
        if pd.isna(motif):
            continue

        # true positives = rows in this motif with a caseid
        tp_rows = g[g["motif_id"].notna()]
        tp = len(tp_rows)

        # average discovered motif length
        mlen = g["motif_length"].dropna().mean() if tp > 0 else 0.0
        
        caseids_in_motif = tp_rows["caseid"].unique()
        required_total = (
            df.loc[df["caseid"].isin(caseids_in_motif), "total_occurances"]
            .dropna()
            .max()
            if len(caseids_in_motif) > 0 else 0
        )

        fn = max(required_total - tp, 0)

        # ---------------------------------------------
        # NEW FALSE-POSITIVE LOGIC (cluster contamination)
        # ---------------------------------------------
        fp = 0
        cluster_ids = tp_rows["cluster_id"].dropna().unique()

        for cid in cluster_ids:
            cluster_group = df[df["cluster_id"] == cid]
            fp += cluster_group[cluster_group["motif"] != motif].shape[0]

        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        if "caseid" not in g.columns:
            caseid = "N/A"
        else:
            caseid = g["caseid"].unique()
        results.append([motif, caseid, mlen, precision, recall, f1, tp, fp, fn, required_total])

    unmapped = df[df["motif"].isna()].shape[0]
    results.append(["UNMAPPED","-", 0.0, 0.0, 0.0, 0, unmapped, 0, 0])    

    return pd.DataFrame(
        results,
        columns=["motif","caseid","average_discovered_motif_length", "precision", "recall", "f1", "tp", "fp", "fn", "required_total"]
    )

def purity_per_two_columns(df, cluster_by_col="caseid", purity_col_name="cluster_id"):
    df_local = df.copy()
    df_local[purity_col_name] = df_local[purity_col_name].astype("object")

    def purity(g):
        counts = g[purity_col_name].value_counts(dropna=False)
        return counts.max() / len(g)

    return (
        df_local
        .groupby(cluster_by_col)
        .apply(purity, include_groups=False)   # <-- fixes the warning
        .rename("purity")
    )

#########################
#########################
# ---- Visualisation ----
#########################
#########################

def plot_density_curve(log, range_low=0, range_high=-1, column_name="rule_density_count"):
    plt.figure(figsize=(10, 4)) 
    if range_high < 0:
        range_high = len(log[column_name])
    plt.plot(log.index[range_low:range_high], log.iloc[range_low:range_high][column_name], linewidth=2) 
    plt.title("Rule Density Across Events in Log") 
    plt.xlabel("Index (Event position in log)") 
    plt.ylabel("Rule Density Count") 
    plt.grid(True, alpha=0.3) 
    plt.tight_layout() 
    plt.show()


def plot_density_curve_with_index_set(log, retain_indices, column_name="rule_density_count"):
    """
    Plots the rule density curve and highlights specific indices/segments.

    Used to highlight which parts of the log were retained after filtering.

    Parameters
    ----------  
    log : pd.DataFrame
        DataFrame containing the rule density column.
    retain_indices : list of int
        Indices to highlight on the plot.
    column_name : str, optional
        Name of the column containing rule density values, by default "rule_density_count"
    """
    plt.figure(figsize=(10, 4))

    # full dataset
    x_full = log.index
    y_full = log[column_name]

    # plot full curve in gray
    plt.plot(x_full, y_full, color="gray", linewidth=2)

    # ensure retain_indices is sorted and array-like
    retain_indices = np.array(sorted(retain_indices))

    # overlay retained points/segments in yellow
    x_ret = log.index[retain_indices]
    y_ret = log.iloc[retain_indices][column_name]

    # Option A — scatter highlight  
    plt.scatter(x_ret, y_ret, color="lightblue", s=20, zorder=3)

    # Option B — connect only consecutive retained indices  
    # (breaks whenever indices are not adjacent)
    blocks = np.split(retain_indices, np.where(np.diff(retain_indices) != 1)[0] + 1)
    for block in blocks:
        if len(block) > 1:
            xb = log.index[block]
            yb = log.iloc[block][column_name]
            plt.plot(xb, yb, color="blue", linewidth=3)

    plt.title("Rule Density Across Events in Log (Retained Index Set Highlighted)")
    plt.xlabel("Index (Event position in log)")
    plt.ylabel("Rule Density Count")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_rule_density_distribution(log: pd.DataFrame, col_name: str = "rule_density_count"):
    counts = log[col_name].value_counts().sort_index()

    # percentiles
    p25 = log[col_name].quantile(0.25)
    p50 = log[col_name].quantile(0.50)
    p75 = log[col_name].quantile(0.75)
    p90 = log[col_name].quantile(0.90)
    p99 = log[col_name].quantile(0.99)

    # cumulative distribution
    cumulative = counts.cumsum()

    fig, ax1 = plt.subplots(figsize=(10, 4))

    # bar plot on primary axis
    bars = ax1.bar(counts.index, counts.values)

    # count labels
    for x, y in zip(counts.index, counts.values):
        ax1.text(x, y, str(y), ha='center', va='bottom', fontsize=8)

    # percentile lines
    ax1.axvline(p25, color="red", linestyle="--", label="25th percentile")
    ax1.axvline(p50, color="green", linestyle="--", label="50th percentile (median)")
    ax1.axvline(p75, linestyle="--", label="75th percentile")
    ax1.axvline(p90, color="orange", linestyle="--", label="90th percentile")
    ax1.axvline(p99, color="purple", linestyle="--", label="99th percentile")

    ax1.set_xlabel(col_name)
    ax1.set_ylabel("count")
    ax1.set_title(f"Distribution Count of {col_name} with Percentiles")

    # secondary axis for cumulative distribution
    ax2 = ax1.twinx()
    ax2.plot(counts.index, cumulative)
    ax2.set_ylabel("cumulative count")

    fig.legend(loc="upper right")
    plt.show()

def plot_rule_density_with_highlights(log: pd.DataFrame,
                                      highlight_df: pd.DataFrame,
                                      density_col: str = "rule_density_count",
                                      range_col: str = "original_df_range",
                                      figsize=(10, 4)):
    """
    Visualizes the rule density curve and highlights the motif/range regions.

    Parameters
    ----------
    log : pd.DataFrame
        Log containing the density values.
    highlight_df : pd.DataFrame
        DataFrame containing ranges in the column `range_col`,
        where each value is a list of integer indices.
    density_col : str
        Column name for the rule density count in `log`.
    range_col : str
        Column name in highlight_df that stores the index ranges (lists of ints).
    figsize : tuple
        Size of the plot.
    """

    plt.figure(figsize=figsize)

    # main curve
    plt.plot(log.index, log[density_col], linewidth=2)

    # highlighted regions
    for g in highlight_df[range_col].tolist():
        if len(g) > 0:
            plt.axvspan(g[0], g[-1], color="red", alpha=0.3)

    plt.title("Rule Density Curve — Highlighted Max Density Regions")
    plt.xlabel("Index (Event position in log)")
    plt.ylabel("Rule Density Count")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()