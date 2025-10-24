import pandas as pd
from collections import Counter,defaultdict
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt


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

def re_pair(log: pd.DataFrame):
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
        print(len(encoding_df), " unique pairs found so far. Level: ", j)
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

import itertools

def find_max_density_groups(
    log, 
    column="rule_density_count", 
    relative_threshold=None, 
    absolute_threshold=None,
    merge_gap=None,
):
    """
    Find indices of maximum (or near-maximum) rule density and group consecutive ones.

    Parameters
    ----------
    log : pd.DataFrame
        DataFrame containing the rule density column.
    column : str, default="rule_density_count"
        Name of the column with rule density values.
    relative_threshold : float, optional
        Include all values >= (relative_threshold * max_density).
        For example, 0.9 means include all densities >= 90% of the max.
    absolute_threshold : float or int, optional
        Include all values >= (max_density - absolute_threshold).
        For example, 1 means include all densities within 1 count of the max.

    Returns
    -------
    max_density : float
        The maximum density value.
    groups : list of lists
        Each inner list contains consecutive indices meeting the threshold.
    """

    # Step 1: find maximum density
    max_density = log[column].max()

    # Step 2: determine cutoff based on threshold
    if relative_threshold is not None:
        cutoff = max_density * relative_threshold
    elif absolute_threshold is not None:
        cutoff = max_density - absolute_threshold
    else:
        cutoff = max_density

    # Step 3: get indices meeting the cutoff
    valid_indices = log.index[log[column] >= cutoff].tolist()

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
    
    return max_density, groups


def find_overlaps(max_groups: list, ground_truth: pd.DataFrame):
    """
    Input:
        max_groups: List of list of events that are close together in rule count
        ground_truth: Dataframe that contains the case_id, start and end index and length of all existing motifs
    
    Returns:
        List of insert ranges and density ranges
    """
    overlaps = []
    for _, row in ground_truth.iterrows():
        s1, e1 = row["start_index"], row["end_index"]
        for g in max_groups:
            s2, e2 = g[0], g[-1]
            if not (e1 < s2 or e2 < s1):  # overlap condition
                overlaps.append({"insert_range": (s1, e1), "density_range": (s2, e2)})
    
    return overlaps

def match_motifs_to_gt(max_groups, ground_truth):
    if not {"start_index", "end_index"}.issubset(ground_truth.columns):
        raise ValueError("ground_truth must have 'start_index' and 'end_index' columns")

    pairs = []
    ground_truth["discovery_count"] = 0

    for mi, g in enumerate(max_groups):
        ms, me = g[0], g[-1]
        for gi in range(len(ground_truth)):
            gs = ground_truth.iloc[gi]["start_index"]
            ge = ground_truth.iloc[gi]["end_index"]
            inter = max(0, min(me, ge) - max(ms, gs) + 1)
            if inter > 0:
                pairs.append((mi, gi, inter))
                ground_truth.at[gi, "discovery_count"] += 1  # increment count

    # Greedy one-to-one assignment (largest overlap first)
    pairs.sort(key=lambda x: x[2], reverse=True)
    matched_motifs, matched_gt = set(), set()
    matched = []
    for mi, gi, _ in pairs:
        if mi not in matched_motifs and gi not in matched_gt:
            matched_motifs.add(mi)
            matched_gt.add(gi)
            matched.append((mi, gi))

    tp = len(matched)
    fp = len(max_groups) - tp
    fn = len(ground_truth) - tp
    return matched, tp, fp, fn, ground_truth


def motif_overlap_metrics(max_groups, ground_truth):
    """
    Compute motif-to-ground-truth overlap statistics.

    Parameters
    ----------
    max_groups : list of lists
        Each element is a list [start_index, ..., end_index] for an identified motif group.
    ground_truth : pd.DataFrame
        Must contain columns ['start_index', 'end_index'].

    Returns
    -------
    dict with:
        - intersection_ratio / intersection_abs
        - undercount_ratio / undercount_abs
        - over_detection_ratio / over_detection_abs
    """

    intersection_ratios, intersection_abs = [], []
    undercount_ratios, undercount_abs = [], []
    overdet_ratios, overdet_abs = [], []

    # --- Intersection (motif overlap with ground truth) ---
    for g in max_groups:
        s2, e2 = g[0], g[-1]
        g_len = e2 - s2 + 1
        overlap_len = 0
        for _, row in ground_truth.iterrows():
            s1, e1 = row["start_index"], row["end_index"]
            inter = max(0, min(e1, e2) - max(s1, s2) + 1)
            overlap_len += inter
        intersection_abs.append(overlap_len)
        intersection_ratios.append(overlap_len / g_len if g_len > 0 else 0)

        # Over-detection: motif part outside GT
        over_len = g_len - overlap_len
        overdet_abs.append(over_len)
        overdet_ratios.append(over_len / g_len if g_len > 0 else 0)

    # --- Undercount (ground truth not covered by motifs) ---
    for _, row in ground_truth.iterrows():
        s1, e1 = row["start_index"], row["end_index"]
        gt_len = e1 - s1 + 1
        overlap_len = 0
        for g in max_groups:
            s2, e2 = g[0], g[-1]
            inter = max(0, min(e1, e2) - max(s1, s2) + 1)
            overlap_len += inter
        undercount_abs.append(max(0, gt_len - overlap_len))
        undercount_ratios.append(1 - (overlap_len / gt_len if gt_len > 0 else 0))

    # --- Aggregate ---
    return {
        "intersection_ratio": np.mean(intersection_ratios) if intersection_ratios else 0,
        "intersection_abs": np.mean(intersection_abs) if intersection_abs else 0,
        "undercount_ratio": np.mean(undercount_ratios) if undercount_ratios else 0,
        "undercount_abs": np.mean(undercount_abs) if undercount_abs else 0,
        "over_detection_ratio": np.mean(overdet_ratios) if overdet_ratios else 0,
        "over_detection_abs": np.mean(overdet_abs) if overdet_abs else 0,
    }


##########################
###### Valmod Code #######
##########################

def remove_redundant_groups(max_groups, log, column="symbol"):
    col = log[column].to_numpy()
    symbol_groups = [[col[i] for i in group] for group in max_groups]

    # group indexes and count
    group_dict = defaultdict(list)
    for idx, g in enumerate(symbol_groups):
        group_dict[tuple(g)].append(idx)

    # build dataframe
    rows = []
    for seq, idxs in group_dict.items():
        rows.append({
            "sequence": list(seq),
            "occurrence": len(idxs),
            "first_occurrence": max_groups[0]
        })

    df = pd.DataFrame(rows)
    return df

def test_multi_threshold_scores(ground_truth, log):
    thresholds = np.arange(0.05, 1.01, 0.05)
    precisions, recalls, f1_scores = [], [], []

    for t in thresholds:
        max_rule_density_count, max_groups = find_max_density_groups(log, relative_threshold=t)
        matched, tp, fp, fn, gt1 = match_motifs_to_gt(max_groups, ground_truth)

        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    # ---- Plot curves ----
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, precisions, marker="o", label="Precision")
    plt.plot(thresholds, recalls, marker="s", label="Recall")
    plt.plot(thresholds, f1_scores, marker="^", label="F1-score")

    plt.axhline(y=1.0, color="red", linestyle="--", linewidth=1)

    plt.title("Precision / Recall / F1 vs. Threshold")
    plt.xlabel("Relative Threshold")
    plt.ylabel("Score")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()