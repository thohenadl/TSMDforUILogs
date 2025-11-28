import sys
sys.path.append('../') # To import from parent dir
import os

import time

import pandas as pd
import numpy as np
import ast

from gensim.models import Word2Vec
from sklearn.preprocessing import LabelEncoder

import util.GrammarBasedUtil as grammar_util
import util.ui_stump as ui_stump

import util.valmod_uihe as valmod_util
from util.util import encoding_UiLog, read_data_for_processing
import matplotlib as plt
import matplotlib.pyplot as plt
import seaborn as sns

from itertools import chain

# LOCOmotif multi variate variable length motif discovery
# Source: https://github.com/ML-KULeuven/locomotif/tree/main
# Paper:  https://link.springer.com/article/10.1007/s10618-024-01032-z
import locomotif.locomotif as locomotif 
import util.locomotif_vis as visualize

def experiment_for_log(log_name: str, encoding_method: int):
    # Set Experiment Parameters To True
    isSmartRPA2025 = True

    # Set Remaining Experiment Parameters To False
    isSmartRPA2024 = False
    isActionLogger = False
    leno_plus = False 
    isRealWorldTest = False
    isHCI = False
    
    # Set Thresholds
    rule_density_threshold = 0.8 # Threshold for Rule Density in Re-Pair >> Grammar Rule Peak is extended left and right until the rule count density drops below this threshold
    stumpy_discovery_threshold = 0.5 # Threshold for STUMPY discovery as a true positiv when overlapping at least this much with a ground truth motif
    app_switch_similarity_threshold = 0.75 # Threshold for considering a pattern drop in all discovered motifs as a pattern switch >> If less than x % share the same app, we consider this a switch
    
    log_name_smartRPA = "log_motifs2_occurances10_length20_percentage25_shuffle0.2.csv"
    data_for_processing = read_data_for_processing(isSmartRPA2024=False,
                                                    isSmartRPA2025=True,
                                                    isRealWorldTest=False,
                                                    isActionLogger=False,
                                                    leno_plus=False,
                                                    isHCI=False,
                                                    log_name_smartRPA=log_name,
                                                    encoding_method=encoding_method)
    
    # Unpack the returned dictionary
    hierarchy_list = data_for_processing["hierarchy_list"]
    hierarchy_columns = data_for_processing["hierarchy_columns"]
    hierarchy_columns_app_switch = data_for_processing["hierarchy_columns_app_switch"]
    file = data_for_processing["file"]
    log = data_for_processing["log"]
    ground_truth = data_for_processing["ground_truth"]
    ui_log_encoded = data_for_processing["ui_log_encoded"]
    ground_truth_start_list = data_for_processing["ground_truth_start_list"]
    column_identifier = data_for_processing["column_identifier"]

    total_start_time = time.time()
    # Apply the Grammar Based Rule Discovery and print a sample rule tree
    encoding_df, symbols, two_gram_df = grammar_util.re_pair(log)
    repair_time = time.time()
    last_entry = len(encoding_df["new_symbol"])-1
    last_encoding = encoding_df["new_symbol"].iloc[last_entry]
    print("\n Last Encoded Entry: ",last_encoding)
    # Just for visualization purposes
    # decoded_symbol = grammar_util.re_pair_decode_symbol(last_encoding, encoding_df, printing=True)
    log = grammar_util.generate_density_count(encoding_df, log)
    density_count_time = time.time()

    # Find the maximum density groups based on the rule density count
    max_density_groups_from_rules = grammar_util.find_max_density_groups(log,relative_threshold=rule_density_threshold,method="percentile",percentile_threshold=0.90)
    maximum_density_groups_df = pd.DataFrame(columns=["group","processed"])
    maximum_density_groups_df["group"] = max_density_groups_from_rules

    # Add start and end indices to the dataframe
    maximum_density_groups_df["start_index"] = -1
    maximum_density_groups_df["end_index"] = -1
    for i, grammer_motif in maximum_density_groups_df.iterrows():
        maximum_density_groups_df.loc[i, "start_index"] = min(grammer_motif['group'])
        maximum_density_groups_df.loc[i, "end_index"] = max(grammer_motif['group'])
        
    start_indices = maximum_density_groups_df["start_index"].tolist()
    end_indices = maximum_density_groups_df["end_index"].tolist()
    max_density_groups_time = time.time()

    # ---- Visualisation Only ---- No Logic for subsequent processing ----
    ground_truth.sort_values(by=["start_index"], inplace=True)
    motiv = int(ground_truth.iloc[0]["start_index"])
    print(f"Start Index of the first ground truth motif: {motiv}")
    colors = plt.cm.tab10.colors
    length = ground_truth["length"].max()
    ground_truth = ground_truth.astype({'caseid': 'str', 'start_index': 'int', 'length': 'int', 'end_index': 'int'})

    print(f"Maximum rule density count: {log['rule_density_count'].max()}")
    max_length = -1
    min_length = len(log)
    for motif in maximum_density_groups_df["group"]:
        if len(motif) > max_length:
            max_length = len(motif)
        if len(motif) > 0 and len(motif) < min_length:
                min_length = len(motif)
                print(f"Longest identified motif length: {max_length}")
                
    grammar_util.plot_density_curve(log, range_low=0, range_high=min(500, len(log)))
    grammar_util.plot_rule_density_distribution(log, col_name="rule_density_count")

    for i, start in enumerate(ground_truth["start_index"]):
        end = start+length+10
        y = log.iloc[int(start):int(end)]["rule_density_count"].to_numpy()
        x = range(len(y))  # all start at 0
        plt.plot(x, y, color=colors[i % len(colors)], linewidth=2, label=f"Range {start}-{end}")
        
    plt.title("Rule Density Across Ground Truth Motif Locations") 
    plt.xlabel("Relative Index (starts at 0)")
    plt.ylabel("rule_density_count")
    plt.tight_layout()
    plt.show()

    stats_before_loco = grammar_util.evaluate_motifs(maximum_density_groups_df["group"], ground_truth)

    overlap_table = stats_before_loco["overlap_table"]         # DataFrame for inspection
    tp_after_grammar, fp_after_grammar, fn_after_grammar = stats_before_loco["tp"], stats_before_loco["fp"], stats_before_loco["fn"]
    for key, value in stats_before_loco.items():
        if key != "overlap_table" and key != "matched_pairs":
            print(f"{key}: {value:.3f}")

    precision_after_grammer = tp_after_grammar / (tp_after_grammar + fp_after_grammar) if (tp_after_grammar + fp_after_grammar) else 0
    recall_after_grammer = tp_after_grammar / (tp_after_grammar + fn_after_grammar) if (tp_after_grammar + fn_after_grammar) else 0
    f1_after_grammar = 2 * precision_after_grammer * recall_after_grammer / (precision_after_grammer + recall_after_grammer) if (precision_after_grammer + recall_after_grammer) else 0

    print(f"\nPrecision: {precision_after_grammer:.3f}, Recall: {recall_after_grammer:.3f}, F1: {f1_after_grammar:.3f}")

    # ---- Plotting the results ----
    plt.figure(figsize=(10, 4))
    plt.plot(log.index, log["rule_density_count"], linewidth=2)
    for g in max_density_groups_from_rules:
        plt.axvspan(g[0], g[-1], color="red", alpha=0.3)
    plt.title("Rule Density Curve — Highlighted Max Density Regions")
    plt.xlabel("Index (Event position in log)")
    plt.ylabel("Rule Density Count")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Extend until the app changes for the specific pattern and add as "lower_app_switch" and "upper_app_switch"
    max_groups_df = grammar_util.app_switch_miner(log, maximum_density_groups_df, hierarchy_columns_app_switch)
    app_switch_mining_time = time.time()

    lower_app_switches = max_groups_df["lower_app_switch"].tolist()
    upper_app_switches = max_groups_df["upper_app_switch"].tolist()

    # Mine the similar paths up and down until an app switch across discovered patterns is found
    result_df = grammar_util.similar_path_up_down(
        df=log,
        max_groups_df=max_groups_df,
        start_indices=lower_app_switches,
        end_indices=upper_app_switches,
        cols=hierarchy_columns_app_switch,
        min_pairs=len(lower_app_switches)*app_switch_similarity_threshold, # At least 75% of the patterns should be similar in path
    )


    # build list of all valid index ranges
    valid_indices = []

    # Include additional safety margin until pattern switch
    range_sum = 0
    for _, row in result_df.iterrows():
        safety_margin = int(row['upper_pattern_switch']) - int(row['lower_pattern_switch']) 
        valid_indices.extend(range(max(0, int(row['lower_pattern_switch'])-safety_margin),
                                min(int(row['upper_pattern_switch'])+safety_margin, len(log)-1)))
        range_distance =  min(int(row['upper_pattern_switch'])+safety_margin, len(log)-1) - max(0, int(row['lower_pattern_switch'])-safety_margin)
        range_sum += range_distance
    print(f"Total Included Range Length: {range_sum} of {len(log)}")
    print(f"Percentage of Log Included: {range_sum/len(log)*100:.2f}%")

    log_path_extension_time = time.time()

    # filter log to include only those indices
    filtered_log = log.loc[log.index.intersection(valid_indices)].copy()

    # optionally, keep the original index as a column for traceability
    filtered_log['original_index'] = filtered_log.index
    filtered_log.reset_index(drop=True, inplace=True)

    # Encode the filtered log
    if encoding_method == 1:
        print("Using Word2Vec based encoding for UI Log")
        filtered_log_encoded = valmod_util.encode_word2vec(filtered_log, orderedColumnsList=hierarchy_columns, vector_size=len(hierarchy_columns)*2)
        column_identifier = 'w2v_'
    elif encoding_method == 2:
        print("Using Hierarchical based encoding for UI Log")
        filtered_log_encoded = encoding_UiLog(filtered_log,orderedColumnsList=hierarchy_columns,encoding=1)
        column_identifier = 'tuple:id'
    elif encoding_method == 3:
        print("Using Co-Occurrance based encoding for UI Log")
        filtered_log_encoded = encoding_UiLog(filtered_log,orderedColumnsList=hierarchy_columns,encoding=2)
        column_identifier = 'tuple:id'
    else:
        raise ValueError("Invalid encoding method selected. Choose 1, 2, or 3.")


    df_filtered_for_plotting = filtered_log.reset_index(drop=True)
    print("Rule Density Curve before Filtering:")
    grammar_util.plot_density_curve(log, range_low=0, range_high=min(500, len(log)))
    print("Rule Density Curve after Filtering:")
    grammar_util.plot_density_curve(df_filtered_for_plotting, range_low=0, range_high=min(500, len(df_filtered_for_plotting)))

    # Reduce the data to an ordered time series for locomotif
    filtered_columns_reduced_log = filtered_log_encoded.filter(like=column_identifier)

    # Optional: Normalization
    ts = (filtered_columns_reduced_log - np.mean(filtered_columns_reduced_log, axis=None)) / np.std(filtered_columns_reduced_log, axis=None)

    # Variable Length Motif Discovery
    motif_sets = locomotif.apply_locomotif(filtered_columns_reduced_log, l_min=5, l_max=65, rho=0.9)
    locomotif_discovery_time = time.time()

    # Plotting with adjusted locomotif visualization (utils)
    fig, ax = visualize.plot_motif_sets(series=filtered_columns_reduced_log.values,
                                        motif_sets=motif_sets, 
                                        max_plots=5,
                                        legend=False)
    plt.show()

    blocks = []

    for cluster_set in motif_sets:# Efficient slicing instead of isin()
        for cluster_motif in cluster_set[1:]:
            for motif in cluster_motif:
                start_of_discovered_motif = motif[0]
                end_of_discovered_motif = motif[1]
                length_of_discovered_motif = end_of_discovered_motif - start_of_discovered_motif
                motif_original_start_index = filtered_log.loc[start_of_discovered_motif, "original_index"]
                # print(f"Motif Start Index: {start}, Motif Length: {end}")
                # print(f"Motif Original Start Index in Full Log: {motif_original_start_index}")

                block = pd.DataFrame({
                    "original_df_range": [list(range(motif_original_start_index, motif_original_start_index+length_of_discovered_motif))],
                    "cluster_id": [cluster_set[0]],
                    "original_df_case_ids": [log.loc[motif_original_start_index:motif_original_start_index+length_of_discovered_motif, "case:concept:name"].tolist()],
                })
                blocks.append(block)


    result_mapped_to_original_index = pd.concat(blocks, ignore_index=True)

    result_mapped_to_original_index = grammar_util.mark_overlaps_grammer_locomotif_indexed(result_mapped_to_original_index, max_groups_df, col_df1="original_df_range", col_df2="group")
    total_end_time = time.time()

    # Filter the result to only include clusters with grammar motif matches
    result_filtered_locomotif_grammer_match = result_mapped_to_original_index.groupby("cluster_id").filter(
        lambda g: g["grammer_motif_match"].any()
    )

    result_filtered_locomotif_grammer_match

    print("Final Evaluation of Discovered Motifs against Ground Truth:")
    print("Considering **all** discovered motifs from LOCOmotif without filtering step")

    final_discovery_result = grammar_util.evaluate_motifs(result_mapped_to_original_index["original_df_range"], ground_truth)

    overlap_table = final_discovery_result["overlap_table"]         # DataFrame for inspection
    tp, fp, fn = final_discovery_result["tp"], final_discovery_result["fp"], final_discovery_result["fn"]
    for key, value in final_discovery_result.items():
        if key != "overlap_table" and key != "matched_pairs":
            print(f"{key}: {value:.3f}")

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    print(f"\nPrecision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")

    print("\nFinal Evaluation of Discovered Motifs against Ground Truth:")
    print("Considering only those motifs that were also identified by Grammar Based Discovery")

    final_discovery_result = grammar_util.evaluate_motifs(result_filtered_locomotif_grammer_match["original_df_range"], ground_truth)

    overlap_table = final_discovery_result["overlap_table"]         # DataFrame for inspection
    tp, fp, fn = final_discovery_result["tp"], final_discovery_result["fp"], final_discovery_result["fn"]
    for key, value in final_discovery_result.items():
        if key != "overlap_table" and key != "matched_pairs":
            print(f"{key}: {value:.3f}")

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")


    # ---- Plotting the results ---- 

    plt.figure(figsize=(10, 4))
    plt.plot(log.index, log["rule_density_count"], linewidth=2)
    for g in result_filtered_locomotif_grammer_match["original_df_range"].to_list():
        plt.axvspan(g[0], g[-1], color="red", alpha=0.3)
    plt.title("Rule Density Curve — Highlighted Max Density Regions")
    plt.xlabel("Index (Event position in log)")
    plt.ylabel("Rule Density Count")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


    


