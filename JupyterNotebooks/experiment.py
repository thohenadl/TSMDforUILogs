import sys
sys.path.append('../') # To import from parent dir
import os

import time

import pandas as pd
pd.set_option("display.max_columns", None)
import numpy as np
import ast
import math

from gensim.models import Word2Vec
from sklearn.preprocessing import LabelEncoder

import util.GrammarBasedUtil as grammar_util
import util.ui_stump as ui_stump

import util.valmod_uihe as valmod_util
from util.util import encoding_UiLog, read_data_for_processing, print_progress_bar
import matplotlib as plt
import matplotlib.pyplot as plt
import seaborn as sns

from itertools import chain

# LOCOmotif multi variate variable length motif discovery
# Source: https://github.com/ML-KULeuven/locomotif/tree/main
# Paper:  https://link.springer.com/article/10.1007/s10618-024-01032-z
import locomotif.locomotif as locomotif 
import util.locomotif_vis as visualize

import warnings
# Example: Only show a warning once globally
warnings.filterwarnings('ignore', category=UserWarning)

def run_experiment(log_name_smartRPA: str, 
                   encoding_method: int=1,
                   rule_density_threshold: float=0.8, 
                   app_switch_similarity_threshold: float=0.75,
                   safety_margin_factor: int=2,
                   percentile_threshold: float=0.90,
                   rho_LoCoMotif : float=0.7,
                   overlap_threshold: float=0.5,
                   printing: bool=False,
                   plotting: bool=False) -> pd.DataFrame:
    isSmartRPA2024 = False
    isSmartRPA2025 = True
    isRealWorldTest = False
    isActionLogger = False
    leno_plus = False
    isHCI = False


    ###########################################################
    #### Experiment Step 1: Data Loading and Preprocessing ####
    ###########################################################
    data_for_processing = read_data_for_processing(isSmartRPA2024=isSmartRPA2024,
                                                    isSmartRPA2025=isSmartRPA2025,
                                                    log_name_smartRPA=log_name_smartRPA)

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

   # Filter out hierarchy columns that have zero unique values
    hierarchy_columns = [
        col for col in hierarchy_columns
        if log[col].nunique() != 0
    ]
    tokens = 0
    for col in hierarchy_columns:
        tokens += log[col].nunique()
    token_based_vector_size = round(math.sqrt(tokens))

    total_start_time = time.time()
    # Apply the Grammar Based Rule Discovery and print a sample rule tree
    encoding_df, symbols, two_gram_df = grammar_util.re_pair(log)
    repair_time = time.time()
    last_entry = len(encoding_df["new_symbol"])-1
    last_encoding = encoding_df["new_symbol"].iloc[last_entry]
    if printing:
        print("\n Last Encoded Entry: ",last_encoding)
    # Just for visualization purposes
    # decoded_symbol = grammar_util.re_pair_decode_symbol(last_encoding, encoding_df, printing=True)
    log = grammar_util.generate_density_count(encoding_df, log)
    density_count_time = time.time()

    ###########################################################
    #### Experiment Step 2: Grammar Based Rule Discovery ######
    ###########################################################

    # Find the maximum density groups based on the rule density count
    max_density_groups_from_rules, _ = grammar_util.find_max_density_groups(log,relative_threshold=rule_density_threshold,method="percentile",percentile_threshold=percentile_threshold)
    maximum_density_groups_df = pd.DataFrame(columns=["group","processed"])
    maximum_density_groups_df["group"] = max_density_groups_from_rules

    # Add start and end indices to the dataframe
    maximum_density_groups_df["start_index"] = -1
    maximum_density_groups_df["end_index"] = -1
    maximum_density_groups_df["length"] = 0
    for i, grammer_motif in maximum_density_groups_df.iterrows():
        maximum_density_groups_df.loc[i, "start_index"] = min(grammer_motif['group'])
        maximum_density_groups_df.loc[i, "end_index"] = max(grammer_motif['group'])
        maximum_density_groups_df.loc[i, "length"] = maximum_density_groups_df.loc[i, "end_index"] - maximum_density_groups_df.loc[i, "start_index"] + 1
        
    start_indices = maximum_density_groups_df["start_index"].tolist()
    end_indices = maximum_density_groups_df["end_index"].tolist()
    max_density_groups_time = time.time()

    # ---- Visualisation Only ---- No Logic for subsequent processing ----
    ground_truth.sort_values(by=["start_index"], inplace=True)
    motiv = int(ground_truth.iloc[0]["start_index"])
    if printing:
        print(f"Start Index of the first ground truth motif: {motiv}")
    ground_truth = ground_truth.astype({'caseid': 'str', 'start_index': 'int', 'length': 'int', 'end_index': 'int'})

    if printing:
        print(f"Maximum rule density count: {log['rule_density_count'].max()}")
        max_length = -1
        min_length = len(log)
        for motif in maximum_density_groups_df["group"]:
            if len(motif) > max_length:
                max_length = len(motif)
            if len(motif) > 0 and len(motif) < min_length:
                min_length = len(motif)
                print(f"Longest identified motif length: {max_length}")

    if plotting:
        grammar_util.plot_density_curve(log, range_low=0, range_high=min(500, len(log)))
        grammar_util.plot_rule_density_distribution(log, col_name="rule_density_count")

    if plotting:
        colors = plt.cm.tab10.colors
        length = ground_truth["length"].max()
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

        # ---- Evaluate the discovered motifs against the ground truth ----
        stats_new = grammar_util.evaluate_motifs(maximum_density_groups_df["group"], ground_truth, overlap_threshold=3, overlap_type="absolute")

        overlap_table = stats_new["overlap_table"]         # DataFrame for inspection
        tp, fp, fn = stats_new["tp"], stats_new["fp"], stats_new["fn"]
        for key, value in stats_new.items():
            if key != "overlap_table" and key != "matched_pairs":
                print(f"{key}: {value:.3f}")

        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

        print(f"\nMetrics after Grammar >> Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")

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

    ########################################################
    ##### App Switch, Pattern Mining and Safety Margin #####
    ########################################################

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

    # Include additional safety margin from pattern switch
    range_sum = 0
    max_safety_margin = 0
    for _, row in result_df.iterrows():
        # Set the safety margin as the distance between the pattern switches & apply factor to extend based on safety concern
        safety_margin = (int(row['upper_pattern_switch']) - int(row['lower_pattern_switch']))*safety_margin_factor
        if safety_margin > max_safety_margin:
            max_safety_margin = safety_margin
        valid_indices.extend(range(max(0, int(row['lower_pattern_switch'])-safety_margin),
                                min(int(row['upper_pattern_switch'])+safety_margin, len(log)-1)))
        range_distance =  min(int(row['upper_pattern_switch'])+safety_margin, len(log)-1) - max(0, int(row['lower_pattern_switch'])-safety_margin)
        range_sum += range_distance
    if printing:
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
        if printing:
            print("Using Word2Vec based encoding for UI Log")
        filtered_log_encoded = valmod_util.encode_word2vec(filtered_log, 
                                                           orderedColumnsList=hierarchy_columns, 
                                                           vector_size=token_based_vector_size,
                                                           completeCorpusLog=log)
        column_identifier = 'w2v_'
    elif encoding_method == 2:
        if printing:
            print("Using Hierarchical based encoding for UI Log")
        filtered_log_encoded = encoding_UiLog(filtered_log,orderedColumnsList=hierarchy_columns,encoding=1)
        column_identifier = 'tuple:id'
    elif encoding_method == 3:
        if printing:
            print("Using Co-Occurrance based encoding for UI Log")
        filtered_log_encoded = encoding_UiLog(filtered_log,orderedColumnsList=hierarchy_columns,encoding=2)
        column_identifier = 'tuple:id'
    else:
        raise ValueError("Invalid encoding method selected. Choose 1, 2, or 3.")


    df_filtered_for_plotting = filtered_log.reset_index(drop=True)
    if plotting:
        print("Rule Density Curve before Filtering:")
        grammar_util.plot_density_curve(log, range_low=0, range_high=min(500, len(log)))
        print("Rule Density Curve after Filtering:")
        grammar_util.plot_density_curve(df_filtered_for_plotting, range_low=0, range_high=min(500, len(df_filtered_for_plotting)))

    # Reduce the data to an ordered time series for locomotif
    filtered_columns_reduced_log = filtered_log_encoded.filter(like=column_identifier)

    # Optional: Normalization
    # ts = (filtered_columns_reduced_log - np.mean(filtered_columns_reduced_log, axis=None)) / np.std(filtered_columns_reduced_log, axis=None)

    ##########################################################
    ##### Variable Length Motif Discovery with LOCOmotif #####
    ##########################################################

    # Calculate length statistics:
    # l_min = Grammer Rule Density Length Mean - Std Dev
    # l_max = Safety Margin from App switch analysis
    l_min = max(5,maximum_density_groups_df["length"].mean() - maximum_density_groups_df["length"].std())
    l_max = max_safety_margin

    # Variable Length Motif Discovery >> Using the low from the GRAMMAR as l_min, Using the security margin from app switch as l_max
    motif_sets = locomotif.apply_locomotif(filtered_columns_reduced_log, l_min=l_min, l_max=l_max, rho=rho_LoCoMotif)
    locomotif_discovery_time = time.time()

    if plotting:
        # Plotting with adjusted locomotif visualization (utils)
        fig, ax = visualize.plot_motif_sets(series=filtered_columns_reduced_log.values,
                                            motif_sets=motif_sets, 
                                            max_plots=5,
                                            legend=False)
        plt.show()

    ###########################################################
    ##### Mapping Motifs back to Original Log Indices #########
    ###########################################################

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
                try:
                    
                    block = pd.DataFrame({
                        "original_df_range": [list(range(motif_original_start_index, motif_original_start_index+length_of_discovered_motif))],
                        "cluster_id": [cluster_set[0]],
                        "original_df_case_ids": [log.loc[motif_original_start_index:motif_original_start_index+length_of_discovered_motif, "case:concept:name"].tolist()],
                    })
                except KeyError as ke:
                    block = pd.DataFrame({
                        "original_df_range": [list(range(motif_original_start_index, motif_original_start_index+length_of_discovered_motif))],
                        "cluster_id": [cluster_set[0]],
                        "original_df_case_ids": [log.loc[motif_original_start_index:motif_original_start_index+length_of_discovered_motif, "caseid"].tolist()],
                    })
                blocks.append(block)
    
    result_mapped_to_original_index = pd.concat(blocks, ignore_index=True)

    result_mapped_to_original_index = grammar_util.mark_overlaps_grammer_locomotif_indexed(result_mapped_to_original_index, max_groups_df, col_df1="original_df_range", col_df2="group")
    total_end_time = time.time()

    #########################################
    #### Discovery before Grammer Filter ####
    #########################################

    if plotting:
        print("Final Evaluation of Discovered Motifs against Ground Truth:")
        print("Considering **all** discovered motifs from LOCOmotif without filtering step")

        final_discovery_result = grammar_util.evaluate_motifs(result_mapped_to_original_index["original_df_range"], ground_truth, overlap_threshold)

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

    ########################################
    #### Discovery after Grammer Filter ####
    ########################################

    # Filter the result to only include clusters with grammar motif matches

    motif_df = result_mapped_to_original_index[
        result_mapped_to_original_index["grammer_motif_match"] == True
    ].copy()

    motif_df = motif_df.reset_index(drop=True)   # index = motif_id = 0..M-1

    # Call evaluate_motifs with the aligned motif_df
    final_discovery_result = grammar_util.evaluate_motifs(
        motif_df["original_df_range"],
        ground_truth,
        overlap_threshold=overlap_threshold,
        overlap_type="ratio"
    )

    overlap_table = final_discovery_result["overlap_table"]         # DataFrame for inspection
    total_tp, total_fp, total_fn = final_discovery_result["tp"], final_discovery_result["fp"], final_discovery_result["fn"]
    # "intersection_ratio": float(np.mean(intersection_ratio)) if M > 0 else 0,
    #     "intersection_abs": float(np.mean(intersection_abs)) if M > 0 else 0,
    #     "undercount_ratio": float(np.mean(under_ratio)) if G > 0 else 0,
    #     "undercount_abs": float(np.mean(under_abs)) if G > 0 else 0,
    #     "over_detection_ratio": float(np.mean(overdet_ratio)) if M > 0 else 0,
    #     "over_detection_abs": float(np.mean(overdet_abs)) if M > 0 else 0,
    total_intersection_ratio = final_discovery_result["intersection_ratio"]
    total_intersection_abs = final_discovery_result["intersection_abs"] 
    total_undercount_ratio = final_discovery_result["undercount_ratio"]
    total_undercount_abs = final_discovery_result["undercount_abs"]
    total_over_detection_ratio = final_discovery_result["over_detection_ratio"]
    total_over_detection_abs = final_discovery_result["over_detection_abs"]
    if printing:
        for key, value in final_discovery_result.items():
            if key != "overlap_table" and key != "matched_pairs":
                print(f"{key}: {value:.3f}")
    
    total_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0
    total_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0
    total_f1 = 2 * total_precision * total_recall / (total_precision + total_recall) if (total_precision + total_recall) else 0

    if printing:
        print(f"Totals >> Precision: {total_precision:.3f}, Recall: {total_recall:.3f}, F1: {total_f1:.3f}")


    # ---- Plotting the results ---- 

    if plotting:
        plt.figure(figsize=(10, 4))
        plt.plot(log.index, log["rule_density_count"], linewidth=2)
        for g in result_mapped_to_original_index[result_mapped_to_original_index["grammer_motif_match"] == True]["original_df_range"].to_list():
            plt.axvspan(g[0], g[-1], color="red", alpha=0.3)
        plt.title("Rule Density Curve — Highlighted Max Density Regions")
        plt.xlabel("Index (Event position in log)")
        plt.ylabel("Rule Density Count")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    cluster_based_evaluation_df = grammar_util.join_discovery_with_ground_truth(final_discovery_result, ground_truth, motif_df)
    if isSmartRPA2025: 
    # For Smart RPA 2025, we have case IDs in the format "motif[n]_caseid" in the ground_truth, which is the motif id in the Leno logs
        cluster_based_evaluation_df["motif"] = cluster_based_evaluation_df["motif_number"]
    motif_res = grammar_util.motif_level_metrics(cluster_based_evaluation_df)
    motif_purity = grammar_util.purity_per_two_columns(cluster_based_evaluation_df, cluster_by_col="motif", purity_col_name="cluster_id")
    if printing:
        print("Motif Level Metrics:")
        print(motif_res)
        print(motif_purity)
        print("\nCluster Level Metrics:")
    cluster_res = grammar_util.cluster_level_metrics(cluster_based_evaluation_df)
    cluster_purity = grammar_util.purity_per_two_columns(cluster_based_evaluation_df, cluster_by_col="cluster_id", purity_col_name="motif")
    if printing:
        print(cluster_res)
        print(cluster_purity)

    # The times that are necessary to execute the discovery steps
    # Works only if executed in sequence as above and with ">> Run All" cells 
    if printing:
        print("\n--- Timing Summary (in seconds) ---")
        print(f"Total Execution Time: {total_end_time - total_start_time:.2f} seconds")
        print(f"Re-Pair Time: {repair_time - total_start_time:.2f} seconds")
        print(f"Density Count Time: {density_count_time - repair_time:.2f} seconds")
        print(f"Max Density Groups Time: {max_density_groups_time - density_count_time:.2f} seconds")
        print(f"App Switch Mining Time: {log_path_extension_time - app_switch_mining_time:.2f} seconds")
        print(f"LOCOmotif Discovery Time: {locomotif_discovery_time - log_path_extension_time:.2f} seconds")

    base_result = {
        "rule_density_threshold": rule_density_threshold,
        "app_switch_similarity_threshold": app_switch_similarity_threshold,
        "encoding_method": encoding_method,
        "safety_margin_factor": safety_margin_factor,
        "total_time": total_end_time - total_start_time,
        "repair_time": repair_time - total_start_time,
        "density_count_time": density_count_time - repair_time,
        "max_density_groups_time": max_density_groups_time - density_count_time,
        "app_switch_mining_time": log_path_extension_time - app_switch_mining_time,
        "locomotif_discovery_time": locomotif_discovery_time - log_path_extension_time,
        "filtered_log_length": len(filtered_log),
        "number_of_discovered_motifs": len(motif_df),
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "total_precision": total_precision,
        "total_recall": total_recall,
        "total_f1": total_f1,
        "total_intersection_ratio": total_intersection_ratio,
        "total_intersection_abs": total_intersection_abs,
        "total_undercount_ratio": total_undercount_ratio,
        "total_undercount_abs": total_undercount_abs,
        "total_over_detection_ratio": total_over_detection_ratio,
        "total_over_detection_abs": total_over_detection_abs,
    }

    # Create a dictionary to store the flattened motif metrics
    flattened_motif_metrics = {}

    for index, row in motif_res.iterrows():
        motif_name = row['motif']
        # If the motif name is an integer, format it as "motifX", otherwise use its string value directly
        # This handles both numeric motifs and "UNMAPPED"
        motif_prefix = f"motif{int(motif_name)}" if isinstance(motif_name, (int, float)) and not pd.isna(motif_name) else str(motif_name).lower()

        # Included caseid for reference
        for col in ['caseid', 'average_discovered_motif_length','precision', 'recall', 'f1', 'tp', 'fp', 'fn', 'required_total']:
            new_column_name = f"{motif_prefix}-{col}"
            flattened_motif_metrics[new_column_name] = row[col]

    # Combine the base results and flattened motif metrics
    final_result_dict = {**base_result, **flattened_motif_metrics}

    # Convert the combined dictionary to a DataFrame
    # We create a DataFrame from a single row dictionary
    final_df = pd.DataFrame([final_result_dict])
    final_df["uiLogName"] = log_name_smartRPA

    return final_df


def experiment(target_filename, rho: float=0.8, log_limit: int=250001, safety_margin_factor: int=2):
    print("Importing necessary Libraries finished. Start execution.")
    validation_data_path = "../logs/smartRPA/202511-update/validationLogInformation.csv"
    validation_data = pd.read_csv(validation_data_path)
    validation_data.sort_values(by="logLength",inplace=True)
    output_csv_path = "../logs/smartRPA/202511-results/" + target_filename

    if os.path.exists(output_csv_path):
        results_df_collector = pd.read_csv(output_csv_path, sep=',')
    else:
        results_df_collector = pd.DataFrame({"uiLogName": []})

    # Iterate, run experiment, and save after each iteration
    i = 0
    for index, row in validation_data.iterrows():
        log_name_smartRPA = row['uiLogName']
        log_length = row['logLength']
        if log_name_smartRPA in results_df_collector['uiLogName'].values:
            i += 1
            print(f"Skipping already processed log: {log_name_smartRPA}")
            continue  # Skip already processed logs
        elif log_length > log_limit:
            # Will run in OOME error with large logs, skip to avoid long runtimes
            print(f"Log {log_name_smartRPA} is too large ({log_length} events). Skipping processing to avoid long runtimes.")
            df_experiment_result = pd.DataFrame([{"uiLogName": log_name_smartRPA, "error_message": "Log too large, skipped processing."}])
        else:
            # Run the experiment for the current log
            try:
                df_experiment_result = run_experiment( # Ensure run_experiment returns a single-row DataFrame with uiLogName
                    log_name_smartRPA=log_name_smartRPA,
                    printing=False,
                    plotting=False,
                    safety_margin_factor=safety_margin_factor,
                    rho_LoCoMotif=rho,
                    overlap_threshold=0.8
                )
            except Exception as e:
                print(f"Error processing {log_name_smartRPA}: {e}")
                df_experiment_result = pd.DataFrame([{"uiLogName": log_name_smartRPA, "error_message": str(e)}])  # Create an empty DataFrame on error
        
        # Merge current validation_data row with experiment result
        # Convert 'row' Series to a DataFrame to allow merging
        current_row_df = pd.DataFrame([row.to_dict()])
        current_row_df_with_results = pd.merge(current_row_df, df_experiment_result, on='uiLogName', how='left')

        # Append the combined row to the collector DataFrame
        # Use ignore_index=True as we're building it row by row
        results_df_collector = pd.concat([results_df_collector, current_row_df_with_results], ignore_index=True)
        
        # Save to CSV after each iteration
        # Use header=True only for the first save, then False for subsequent appends
        mode = 'w'
        header = True 
        results_df_collector.to_csv(output_csv_path, mode=mode, header=header, index=False)
        
        print(f"Saved results for {log_name_smartRPA} to {output_csv_path}")
        print_progress_bar(i + 1, len(validation_data))
        i += 1

    print("\nFinal DataFrame collected (last state before loop finished):")
    print(results_df_collector)