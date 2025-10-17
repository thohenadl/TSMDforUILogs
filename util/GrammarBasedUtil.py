import pandas as pd
from collections import Counter,defaultdict
import pandas as pd
import itertools
import matplotlib.pyplot as plt

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

def plot_density_curve(log, column_name="rule_density_count"):
    plt.figure(figsize=(10, 4)) 
    plt.plot(log.index, log[column_name], linewidth=2) 
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

def find_max_density_groups(log, column="rule_density_count"):
    """
    Find indices of maximum rule density and group consecutive ones.

    Returns
    -------
    max_density : int or float
        The maximum rule density value.
    groups : list of lists
        Each inner list contains consecutive indices with the maximum density.
    """

    # Step 1: find maximum density value
    max_density = log[column].max()

    # Step 2: collect indices with that value
    max_indices = log.index[log[column] == max_density].tolist()

    # Step 3: group consecutive indices into connected ranges
    groups = []
    for _, group in itertools.groupby(
        enumerate(max_indices), lambda x: x[0] - x[1]
    ):
        connected = [g[1] for g in group]
        groups.append(connected)

    return max_density, groups
