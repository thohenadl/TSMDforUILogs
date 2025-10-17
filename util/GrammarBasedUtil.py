import string
import pandas as pd
from collections import Counter,defaultdict
import pandas as pd

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

def reconstruct_encoding(encoded_list, df, hierarchy_list_columns):
    # keep only first occurrence per symbol in the original df
    first_occ = df.drop_duplicates(subset="symbol", keep="first")

    # make sequence frame
    seq_df = pd.DataFrame({"symbol": encoded_list, "order": range(len(encoded_list))})

    # merge to reconstruct hierarchy attributes
    merged = seq_df.merge(first_occ[["symbol"] + hierarchy_list_columns],
                          on="symbol", how="left")

    # restore order and select hierarchy columns
    reconstructed = merged.sort_values("order")[hierarchy_list_columns].reset_index(drop=True)

    return reconstructed

# ---- Rule Density Curve ----


