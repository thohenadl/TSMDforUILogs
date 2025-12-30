# If a function is added in here the Jupyter Notebook has to be restarted (Kernal restart) to load the function properly

import pandas as pd
import os
import random
import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix

from itertools import chain
import util.valmod_uihe as valmod_util
import util.GrammarBasedUtil as grammar_util

from collections import Counter

from stumpy import config
import stumpy
import ast

from datetime import datetime

# ---- Tuple Generation for Time Series Data Creation ----
def createDict(someSet) -> dict:
    theDict = {}
    id_counter = 0
    for element in someSet:
        theDict[element] = id_counter
        id_counter += 1
    return theDict

# Define a function to retrieve the key from the dictionary
def get_key(row, mapping_dict, row_name):
    value = row[row_name]
    return mapping_dict.get(value)

def get_id(row, tuples, columns):
    # Used to generate one dimensional time series
    try:
        # Get the index of the first occurrence of element_value
        element_index = tuples.index(tuple(row[columns]))
        return element_index
    except ValueError:
        return -1
    
def encoding_UiLog(uiLog: pd.DataFrame, orderedColumnsList: list= ["category","application","concept:name"],
                   encoding: int=1, cooccurance_distance: int=2, coocurance_combined: bool=True) -> pd.DataFrame:
    '''
    Method to encode the UILog based on the selected method. Default Method is continuous hot encoding.
    Default assumption is to encode a SmartRPA generated UI log with the columns "category", "application", and "concept:name".

    Parameters:
      uiLog (pd.DataFrame): Dataframe containing the UI Log
      orderedColumnsList (List / Default SmartRPA Columns): Ordered List of columns containing the attributes from the UI Log
      encoding (Int / Default 1): Encoding Method to be used (1=Hierarchy Encoding, 2=Co-Occurrance Encoding, 3=Hot Ecnoding)
      cooccurance_distance (Int / Default 2): Distance to be considered for the co-occurrance matrix counting
      coocurance_combined (Bool / Default True): If the columns for cooccurance should be combined into a single value

    Returns:
      Encoded UI log containing the column "tuple:id" as encoded value
    '''

    if encoding == 1: # Hierarchy Encoding
      # Get all Unique Combinations
      uniqueDF = uiLog[orderedColumnsList].drop_duplicates()
      # Group them based on hierarchy order
      uniqueDF.groupby(by=orderedColumnsList,as_index=True,dropna=False)
      uniqueDF.sort_values(by=orderedColumnsList,inplace=True,ignore_index=True)
      uniqueDF = uniqueDF.reset_index(names='tuple:id')
      # Merge DF based on SQL Statement
      return pd.merge(uiLog,uniqueDF, how="left", on=orderedColumnsList)
    
    elif encoding == 2: # Co-Occurrance Encoding
      result_df = uiLog.copy()
      if coocurance_combined:

        # Create combined column in result_df
        result_df['combined'] = result_df[orderedColumnsList].astype(str).agg('|'.join, axis=1)
        
        # Create temporary df with combined column for co-occurrence calculation
        temp_df = result_df.copy()
        combined_matrix = co_occurrence_matrix_n(temp_df, cooccurance_distance, 'combined')
        combined_ordered = spectral_ordering_cooccurrence(combined_matrix)
        combined_dict = createDict(list(combined_ordered))
        
        # Apply mapping to result_df using the existing combined column
        result_df['tuple:id'] = result_df['combined'].map(lambda x: combined_dict.get(x))
        
        # Clean up temporary column
        result_df = result_df.drop('combined', axis=1)
        
      else:
        # Does not function very well, looks like continuous hot encoding
        # Process each column individually
        # Process columns hierarchically
        id_columns = []
        ordering_columns = []
        
        for col in orderedColumnsList:
            # Calculate co-occurrence and ordering for this column
            col_matrix = co_occurrence_matrix_n(uiLog, cooccurance_distance, col)
            col_ordered = spectral_ordering_cooccurrence(col_matrix)
            col_dict = createDict(list(col_ordered))
            
            # Add column IDs
            id_col = f"{col}:id"
            order_col = f"{col}:order"
            
            result_df[id_col] = result_df.apply(
                lambda row: get_key(row, col_dict, col), axis=1)
            result_df[order_col] = result_df[id_col].map(
                {val: idx for idx, val in enumerate(sorted(result_df[id_col].unique()))})
            
            id_columns.append(id_col)
            ordering_columns.append(order_col)
        
          # Create hierarchical ordering
        result_df['sequential_order'] = 0
        multiplier = 1
        
        for order_col in reversed(ordering_columns):
            result_df['sequential_order'] += result_df[order_col] * multiplier
            multiplier *= len(result_df[order_col].unique())
        
        # Generate tuples preserving order
        result_df = result_df.sort_values('sequential_order').reset_index(drop=True)
        unique_df = result_df[id_columns].drop_duplicates(keep='first').reset_index(drop=True)
        tuples = [tuple(row[id_columns]) for _, row in unique_df.iterrows()]
        result_df['tuple:id'] = result_df.apply(
            lambda row: get_id(row, tuples, columns=id_columns), axis=1)
      
      return result_df
    
      # Old Code with hard coded columns

      # Encode the application data by Cooccurance
      # application_co_matrix = co_occurrence_matrix_n(uiLog, cooccurance_distance, "application")
      # application_matrix = spectral_ordering_cooccurrence(application_co_matrix)
      # applicationDict = createDict(list(application_matrix))
      # # Encode the concept name (action) by Cooccurance
      # concept_name_co_matrix = co_occurrence_matrix_n(uiLog, cooccurance_distance, "concept:name")
      # concept_name_matrix = spectral_ordering_cooccurrence(concept_name_co_matrix)
      # conceptNamesDict = createDict(list(concept_name_matrix))
      # # Encode the categories with no special order as they are very few
      # categoriesDict = createDict(set(uiLog.sort_values(by=['category'])['category'].unique()))

      # uiLog['application:id'] = uiLog.apply(lambda row: get_key(row, applicationDict, 'application'), axis=1)
      # uiLog['concept:name:id'] = uiLog.apply(lambda row: get_key(row, conceptNamesDict, 'concept:name'), axis=1)
      # uiLog['category:id'] = uiLog.apply(lambda row: get_key(row, categoriesDict, 'category'), axis=1)

      # # Encode all ids into a single value for univariate discovery by using tuples
      # numbersDF = uiLog[['concept:name:id', 'application:id', 'category:id']]

      # # Generate unique tuples for indexing the individual combinations of the rows mentioned
      # unique_df = numbersDF.drop_duplicates(subset=numbersDF.columns, keep='first')
      # tuples = [tuple(row[['concept:name:id', 'application:id', 'category:id']]) for i, row in unique_df.sort_values(by='application:id').iterrows()]
            
      # uiLog['tuple:id'] = uiLog.apply(lambda row: get_id(row, tuples, columns=['concept:name:id','application:id', 'category:id']), axis=1)
    
    elif encoding == 3: # Continuous Hot Encoding
      # Create a single combined column to represent the unique combination
      combined = uiLog[orderedColumnsList].astype(str).agg('|'.join, axis=1)
      
      # Factorize the combined values to get unique IDs
      uiLog['tuple:id'] = pd.factorize(combined)[0]
      return uiLog
    
    else:
      raise ValueError("Encoding method not supported. Please select a valid encoding method.")

# ---- Motif Discovery ----

def discover_motifs(uiLog: pd.DataFrame, window_size: int=25, normalize=True, self_exclude: bool=False):
    """
    Args:
      uiLog (DataFrame): Encoded uiLog containing the columns in text and integer format
      window_size (int): Window Size
      normalize : bool, default True
        When set to ``True``, this z-normalizes subsequences prior to computing
        distances. Otherwise, this function gets re-routed to its complementary
        non-normalized equivalent set in the ``@core.non_normalized`` function
        decorator.
    Returns:
      stumpy tm_matrix
    """
    starting_row = 0
    ending_row = len(uiLog)-1
    #Extract ids and rows
    if self_exclude:
       from stumpy import config
       config.STUMPY_EXCL_ZONE_DENOM = 1  # The exclusion zone is i ± window_size
    event_series = uiLog.loc[starting_row:ending_row,'tuple:id'].values.astype(float)
    tm_matrix = stumpy.stump(T_A=event_series, m=window_size, normalize=normalize)

    return tm_matrix, event_series

def reduceLogToDiscovered(dataframe: pd.DataFrame, topMotifIndex: list, windowSize: int):
    """
    Reduces a pandas DataFrame containing event logs to only the discovered motifs based on provided indices and window size.

    Args:
    - dataframe (pd.DataFrame): The input DataFrame containing the event logs. It's assumed that the DataFrame has a column structure representing the event sequence.
    - topMotifIndex (list): A list of integer indices representing the starting positions of the discovered motifs within the original DataFrame.
    - windowSize (int): The window size used for motif discovery.

    The function iterates through the provided `topMotifIndex` list and extracts the corresponding subsequences from the original DataFrame. Each extracted subsequence represents a discovered motif. To ensure data integrity, the function handles potential out-of-bounds index errors by checking if the start index falls within the DataFrame boundaries. Additionally, the end index is adjusted to avoid exceeding the DataFrame length.
    The extracted subsequences (motifs) are then augmented with a new column named "case:concept:name" containing a unique case identifier. This identifier helps distinguish between different discovered motifs within the resulting DataFrame. Finally, all extracted motifs are concatenated into a new DataFrame and returned.

    Returns:
        pd.DataFrame: A new DataFrame containing only the discovered motifs extracted from the original DataFrame, with an additional column "case:concept:name" for case identification.
    """
    if "case:concept:name" not in dataframe.columns:
      new_df = pd.DataFrame(columns=dataframe.columns.tolist() + ["case:concept:name"])  # Add case column
    else: 
      new_df = pd.DataFrame(columns=dataframe.columns.tolist())
    case_id = 0
    for start_index in topMotifIndex:
      # Ensure start index is within dataframe bounds
      if start_index < 0 or start_index >= len(dataframe):
        continue
      end_index = min(start_index + windowSize, len(dataframe))  # Handle potential out-of-bounds end index
      window_df = dataframe.iloc[start_index:end_index].copy()
      window_df["case:concept:name"] = str(case_id)
      new_df = pd.concat([new_df, window_df], ignore_index=True)
      case_id += 1
    return new_df

# ---- Validation Data Generation ----
def read_csvs_and_combine(folder_path, max_rows=100000):
   """Reads all CSV files in a folder, combines them into a single DataFrame, and stops reading if the limit is reached.

   Args:
       folder_path (str): Path to the folder containing CSV files.
       max_rows (int, optional): Maximum number of rows to read. Defaults to 100000.

   Returns:
       pandas.DataFrame: Combined DataFrame of all read CSV files.
   """
   df = pd.DataFrame()
   for filename in os.listdir(folder_path):
       if filename.endswith(".csv"):
           file_path = os.path.join(folder_path, filename)
           temp_df = pd.read_csv(file_path)
           # Check if appending would exceed the limit
           if len(df) + len(temp_df) > max_rows:
               print(f"Maximum row limit of {max_rows} reached. Stopping reading additional files.")
               break
           # Append to the DataFrame
           df_list = [df,temp_df]
           df = pd.concat(df_list, ignore_index=True)
   return df

def random_n(min_value, max_value):
    """Calculates a random integer between a specified minimum and maximum value (inclusive).
    
    Args:
      min_value (int): The minimum value (inclusive).
      max_value (int): The maximum value (inclusive).
    
    Returns:
      int: A random integer between min_value and max_value.
    
    Raises:
      ValueError: If the minimum value is greater than the maximum value.
    """
    if min_value > max_value:
        raise ValueError("Minimum value cannot be greater than maximum value.")
    
    return random.randint(min_value, max_value)

def select_consecutive_rows(df, n):
    """Selects n random consecutive rows from a DataFrame.
    
    Args:
      df (pd.DataFrame): The DataFrame to select from.
      n (int): The number of consecutive rows to select.
    
    Returns:
      pd.DataFrame: A DataFrame containing the selected rows.
    """
    if n > len(df):
        raise ValueError("n cannot be greater than the length of the DataFrame")
    # Get a random starting index within the valid range
    start_idx = np.random.randint(0, len(df) - n + 1)
    return df.iloc[start_idx:start_idx + n]

def biased_length(n_max, p=0.5):
    # geometric-like draw: repeat until within range
    while True:
        x = np.random.geometric(p)
        if x <= n_max:
            return x
        
def get_rand_uiLog(dfAll: pd.DataFrame, n_max: int, action_count: int, printing=False) -> pd.DataFrame:
    """
    Efficient and scalable generator of noise UI logs.
    Avoids O(n^2) concatenation. Builds blocks then concatenates once.
    """
    blocks = []
    total = 0
    seq_len_counter = Counter()

    max_len = len(dfAll)

    while total < action_count:
        if printing and total % 100 == 0:
            print(f"Current generated UiLog length: {total}")

        seq_len = biased_length(n_max, p=0.35)
        if seq_len >= max_len:
            seq_len = max_len - 1

        start_idx = np.random.randint(0, max_len - seq_len)
        seq = dfAll.iloc[start_idx : start_idx + seq_len]

        blocks.append(seq)
        total += seq_len
        seq_len_counter[seq_len] += 1

    # print summary if printing: 
    if printing:
        print("Sequence length usage:") 
        for length, count in sorted(seq_len_counter.items()): 
            print(f"{length}: {count}")

    result = pd.concat(blocks, ignore_index=True)
    print(f"Generated random UiLog length: {len(result)}")
    return result.iloc[:int(action_count)]

def get_completely_random_uiLog(df, actions=9600):
    """
    Generates a random user interaction (UI) log by sampling rows from the provided DataFrame.

    This function takes two arguments:

    - df (pd.DataFrame): The input DataFrame containing the user interaction log data.
    - actions (int, optional): The desired number of actions in the resulting random UI log. Defaults to 9600.

    Returns:
        pd.DataFrame: A new DataFrame containing a random selection of rows from the original DataFrame, representing a random UI log with the specified number of actions (or the maximum number of possible actions if `actions` exceeds the original DataFrame length).
    """
    if actions <= 0:
      raise ValueError("Actions must be a positive integer")

    # Generate random indices efficiently using numpy.random.randint
    indices = random.sample(range(0, min(len(df)-1,actions)), min(len(df)-1,actions))
    while len(indices)-1 < actions:
      more_indices = random.sample(range(0, min(len(df)-1,actions)), min(len(df)-1,actions))
      indices = indices + more_indices

    # Select rows using efficient indexing and concatenation
    indices = indices[:actions]
    # ui_log = pd.concat([df.iloc[i:i+1] for i in indices])
    ui_log = df.iloc[indices]

    return ui_log

def get_random_values(df: pd.DataFrame, column_name: str, m: int, min_len:int=1):
    """
    Gets r random values from a specified column in a DataFrame.
    
    Args:
      df (pd.DataFrame): The DataFrame to get values from.
      column_name (str): The name of the column containing the desired values.
      m (int): The number of random values to get.
      min_len (int): Minimal length of routine to be found, default 1 action.
    
    Returns:
      list: A list containing the r random values from the specified column.
    """
    # Check if column exists
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in the DataFrame.")
    # Get value counts efficiently using Series.value_counts()
    # Get value counts
    value_counts = df[column_name].value_counts()
    
    # Filter rows based on minimum occurrence
    filtered_df = df[df[column_name].isin(value_counts[value_counts >= min_len].index)] 
    random_values = filtered_df[column_name].sample(m)
    random_cases = pd.DataFrame(columns=[column_name,"count"])
    random_cases[column_name] = random_values
    random_cases["count"] = random_cases[column_name].map(value_counts)
    return random_cases

def reorder_dataframe(df, reorder_percentage=10, inplace=False):
    """
    Reorders a pandas DataFrame for a specified percentage of elements.
    
    Args:
      df (pd.DataFrame): The DataFrame to reorder.
      reorder_percentage (int, optional): The percentage of elements to reorder (defaults to 0.1).
      inplace (bool, optional): Whether to modify the DataFrame in-place (defaults to False).
    
    Returns:
      pd.DataFrame: The reordered DataFrame.
    """
    
    if not 0 <= reorder_percentage <= 100:
        raise ValueError("reorder_percentage must be between 0 and 100.")
    
    if not inplace:
        df = df.copy()  # Create a copy if not modifying in-place
    
    # Get the number of elements to reorder
    num_elements_to_reorder = round(len(df) * (reorder_percentage/100))
    # Randomly select elements to reorder, ensuring they are within valid range (0 to len(df) - 1)
    valid_range = (0, len(df) - 2)
    elements_to_reorder = [random.randint(*valid_range) for _ in range(num_elements_to_reorder)]
    # Shuffle the selected elements within the list
    random.shuffle(elements_to_reorder)
    
    # Reorder elements in the DataFrame
    for i, element_index in enumerate(elements_to_reorder):
        new_position = random.randint(*valid_range)
        df.iloc[[element_index, new_position]] = df.iloc[[new_position, element_index]]
    
    return df

def remove_n_percent_rows(df, n):
    """
    Removes n% of rows from a DataFrame randomly.
    
    Args:
      df (pd.DataFrame): The DataFrame to modify.
      n (int): The percentage of rows to remove (0 to 100).
    
    Returns:
      pd.DataFrame: The modified DataFrame with rows removed.
    """
    
    if not 0 <= n <= 100:
        raise ValueError("n must be between 0 and 100 (inclusive).")
    if n == 0:
        return df.copy()  # Return a copy without modification
    
    # Calculate the number of rows to remove
    num_rows_to_remove = int(len(df) * (n / 100))
    
    # Randomly sample rows to remove
    rows_to_remove = df.sample(num_rows_to_remove, random_state=42)  # Set random state for reproducibility
    
    # Remove the selected rows
    return df.drop(rows_to_remove.index)

def insert_rows_at_random(df: pd.DataFrame, insert_df: pd.DataFrame, o: int, 
                          shuffled:bool=False, shuffled_by:int=10, reduced:bool=False, reduced_by:bool=10):
    """
    Deprecated and replaced with function: insert_motifs_non_overlap
    
    Inserts rows from one DataFrame into another at random positions o times, keeping them together.
    
    Args:
      df (pd.DataFrame): The base DataFrame to insert rows into.
      insert_df (pd.DataFrame): The DataFrame containing the rows to insert.
      o (int): The number of times to insert the rows.
      shuffled (bool): Should the insert_df have sequence change
      shuffled_by (int): Percent the dataframe should be shuffled
      reduced (bool): Should the insert_df be reduced in percent
      reduced_by (int): Percent the dataframe should be reduced
    
    Returns:
      pd.DataFrame: The modified DataFrame with inserted rows.
    """
    # Ensure valid range for random numbers (from 0 to df_length - 1)
    valid_range = (0, len(df.index)-1)
    # Generate random numbers and limit them to the valid range
    index_list = sorted([random.randint(*valid_range) for _ in range(o)])
    for insert_indices in index_list:
        # Ensure to not interrupt a previously inserted routine
        # ToDo
        # Insert the entire insert_df at the chosen index
        if shuffled:
            insert_df = reorder_dataframe(insert_df,shuffled_by)
        if reduced:
            insert_df = remove_n_percent_rows(insert_df,reduced_by)
        df = pd.concat([df.iloc[:insert_indices], insert_df, df.iloc[insert_indices:]], ignore_index=True)
    return df, index_list

def insert_motifs_non_overlap(random_cases_list, uiLog, dfcases, occurances, case_column_name, sorted_insert_col, 
                              shuffled:bool=False, shuffled_by:int=10, reduced:bool=False, reduced_by:int=10):
    """
    random_cases_list (List): n cases that should be taken from all possibles to be inserted
    uiLog (df): Prepared uiLog containing no motifs
    dfcases (df): Dataframe containing the filtered cases
    occurances (int): Number of times the motifs/cases should be added
    case_column_name (str): Name of the case id column
    sorted_insert_col (str): Name of the column to sort the dataframe for insertion
    shuffled (bool): Should the dataframe to be inserted be shuffled
    shuffled_by (int): Percent (as int not float) to shuffle by
    reduced (bool): Should the dataframe to be inserted be reduced
    reduced_by (int): Percent (as int not float) to reduce by
    """
    # Shuffle the cases occurrance times into the list
    random_cases_list = random_cases_list * occurances
    random_cases_list = random_cases_list[:occurances]
    random.shuffle(random_cases_list) # Trimming the list as there can be > random_case_list but we only want occurances of entries
    # print(f"Random Cases List Len: {len(random_cases_list)}; {random_cases_list}")
    
    # Generate random numbers and limit them to the valid range
    # Make them descending to order without random overlap
    index_list = sorted([random.randint(0,len(uiLog)-1) for _ in range(len(random_cases_list))],reverse=True) 
    
    # Inserting the routines top down
    for i, routine in enumerate(random_cases_list):
        # Get the case elements and order by sorted_insert_col (timestamp)
        insert_df = dfcases[dfcases[case_column_name] == random_cases_list[i]].sort_values(sorted_insert_col)
        
        if reduced:
            insert_df = remove_n_percent_rows(insert_df,reduced_by)
        if shuffled and shuffled_by > 0:
            insert_df = reorder_dataframe(insert_df,shuffled_by)
        
        uiLog = pd.concat([uiLog.iloc[:index_list[i]], insert_df, uiLog.iloc[index_list[i]:]], ignore_index=True)
        # Correct the indexes by the length of the inserted dataframe (routine)
        index_list = [x+len(insert_df) for x in index_list[:i]] + index_list[i:]
        
        # For debugging the index list correction
        # print(f"After: Index loop i = {i}, Len UI Log = {len(uiLog)}, random cases list len = {len(random_cases_list)}, indices = {index_list}")
    return uiLog, index_list, random_cases_list

# ---- Window Size Selection ----
def windowSizeByBreak(uiLog: pd.DataFrame, timestamp:str="time:timestamp", realBreakThreshold:float=950.0, percentil:int=75) -> int:
    """
    Calculates the window size based on the average number of actions between major breaks.
    A major break is considered everything above the third percential of breaks in the UI log.
    Major breaks that are not considered are once above the realBreakThreshold in seconds.
    
    Args:
      uiLog (pd.DataFrame): The ui log that should be processed
      timestamp (str): The column name containing the time stamps
      realBreakThreshold (float): Time in seconds for which a break is a business break (e.g. new day, coffee break)
      percentil (int): Percentil, which should be used for seperating
    
    Returns:
      windowSize (int)
    """
    b = 0
    i = 0
    breaks = []
    uiLog[timestamp] = pd.to_datetime(uiLog[timestamp], format='ISO8601')
    # Calculate time differences (assuming timestamps are sorted)
    breaks = uiLog['time:timestamp'].diff().dt.total_seconds().tolist()[1:]
    breaks = [gap for gap in breaks if gap <= realBreakThreshold]
    third_quartile = np.percentile(breaks, percentil)
    # Find indices of third quartile occurrences
    quartile_indices = [i for i, value in enumerate(breaks) if value == third_quartile]

    # Check if there are at least two occurrences
    if len(quartile_indices) < 2:
        return None  # Not enough data to calculate average

    # Calculate the number of elements between occurrences (excluding the quartiles themselves)
    num_elements_between = [quartile_indices[i + 1] - quartile_indices[i] - 1 for i in range(len(quartile_indices) - 1)]

    # Calculate the average number of elements between occurrences
    average_elements = sum(num_elements_between) / len(num_elements_between)
    return third_quartile,quartile_indices,average_elements


# ------ Boundary Information and Evaluation functions -----

# Method to calculate the rolling mean for timeDifferences
def calculate_running_average_difference(df, n:int, upper_boundary:int = 3600, col_name="timeDifference"):
    """
    This function calculates the running average difference between timestamps 
    for the last n events in a pandas dataframe.

    Args:
        df (pandas.DataFrame): The dataframe containing the time difference column.
        n (int): The number of events to consider for the running average.
        col_name (str, optional): The name of the column containing the time differences. Defaults to "timediff".

    Returns:
        pandas.DataFrame: The modified dataframe with a new column named "n-running-difference".
    """
    df["n-running-difference"] = df[col_name].rolling(window=n).apply(lambda x: np.clip(x,a_min=0,a_max=upper_boundary).mean(), raw=True)
    return df

# Adding boundary information if time difference >1h between actions in case
def calculate_time_difference(arr: pd.DataFrame, timeStampCol:str, gap:int = 3600, n_rolling:int = 100) -> pd.DataFrame:
    """
    Calculates the time difference between consequitive rows and sets the timeDifferenceBool flags
    Static gap takes the gap parameter with default value 3600s (1h)
    Dynamic gap takes the n-rolling average gap with n as parameter with default 100

    Args:
      arr (dataframe): The UI log
      miner (uipatternminer): A generated UI pattern miner generated on the Dataframe
      gap (int: def. 3600): Comparison value for gap between two actions, if higher a gap is detected
      n_rolling (int: def. 100): Input value to calculate the dynamic running value 
    """
    arr[timeStampCol] = pd.to_datetime(arr[timeStampCol])

    arr['timeDifference'] = pd.Timedelta(seconds=0) # Initialisiere die Zeitdifferenz-Spalte

    for index, row in arr.iterrows():
        if index < len(arr) - 1:
          time_diff = arr.loc[index + 1, timeStampCol] - row[timeStampCol]
          arr.at[index, 'timeDifference'] = time_diff

    # Setting static gap
    arr['timeDifferenceBoolStatic'] = arr['timeDifference'].apply(lambda x: x.total_seconds() > gap)

    # Setting dynamic gap
    try:
        arr["timeDifference"] = arr.apply(lambda row: row["timeDifference"].seconds, axis=1)
    except:
        nothingToDoHere = 1
        # We actually expect it to be integer values already, otherwise something has gone wrong with the df earlier
    # THe running mean is calculated without considering large gaps, e.g. gaps over 3600s=1h
    arr = calculate_running_average_difference(arr.copy(), n_rolling)
    arr["timeDifferenceBoolRolling"] = arr.apply(lambda row: row['n-running-difference'] < row['timeDifference'], axis=1)

    return arr

def find_closest_boundaries(df, index, col_name='isBoundary'):
    """
    Finds closest forward and backward indices with True value in a column.

    Args:
        df (pd.DataFrame): The DataFrame to search.
        index (int): The index of the reference row.
        col_name (str, optional): The name of the column containing boolean values.
            Defaults to 'isBoundary'.

    Returns:
        tuple: A tuple containing two elements:
            - forward_index (int): Index of the closest row forward with True in 'col_name'.
            - backward_index (int): Index of the closest row backward with True in 'col_name'.

    Raises:
        ValueError: If the index is out of bounds of the DataFrame.
    """

    if index < 0 or index >= len(df):
      raise ValueError("Index out of bounds of the DataFrame.")

    # Forward search (excluding the current index)
    forward_idx = df[index:].loc[df[index:].iloc[:]["isBoundary"] == True].index[0]  # Access first True index

    # Backward search (excluding the current index)
    backward_idx = df[:index].loc[df[:index].iloc[:]["isBoundary"] == True].index[-1]  # Access last True index (reverse order)

    # Handle cases where there's no boundary value forward/backward
    if forward_idx == index:
      forward_idx = None
    if backward_idx == index:
      backward_idx = None

    return forward_idx, backward_idx


# ---- Optimizing Encoding of values -----

def co_occurrence_matrix_n(df: pd.DataFrame, n: int, coOccuranceCol: str):
  """
  Generates a co-occurrence matrix for n values before and after each row in a DataFrame.

  Args:
      df (pd.DataFrame): The DataFrame containing the concept:name column.
      n (int): The number of values to consider before and after the index value.
      coOccuranceCol (str): The column for which the co-occurance should be counted

  Returns:
      pd.DataFrame: The co-occurrence matrix.
  """
  # Add padding values
  padding = ["PAD"] * n
  df_padded = pd.concat([pd.Series(padding), df[coOccuranceCol]], ignore_index=True)
  df_padded = pd.concat([df_padded, pd.Series(padding)], ignore_index=True)

  # Create co-occurrence matrix
  co_matrix = pd.DataFrame(columns=df_padded.unique(), index=df_padded.unique())
  co_matrix.fillna(0, inplace=True)

  # Iterate through rows (excluding padding rows)
  for i in range(n, len(df_padded) - n - 1):
    current = df_padded.iloc[i]
    # Get n values before and after
    previous_values = df_padded.iloc[i-n:i].tolist()
    next_values = df_padded.iloc[i+1:i+n+1].tolist()

    # Increment co-occurrence counts
    for prev in previous_values:
      co_matrix.loc[current, prev] += 1
    for next in next_values:
      co_matrix.loc[current, next] += 1

  # Remove padding rows and columns
  try:
      co_matrix = co_matrix.drop("PAD", axis=1)
  except:
    print("PAD column could not be removed or was not present in dataframe.")
  try:
      co_matrix = co_matrix.drop("PAD", axis=0)
  except:
    print("PAD row could not be removed or was not present in dataframe.")
  
  return co_matrix


def spectral_ordering_cooccurrence(co_matrix):
  """
  Reorders a co-occurrence matrix using spectral ordering.

  Args:
      co_matrix (pd.DataFrame): The co-occurrence matrix.

  Returns:
      pd.DataFrame: The reordered co-occurrence matrix.
  """
  # If there are issues with the ARPACK look here: https://docs.scipy.org/doc/scipy/tutorial/arpack.html
  # Check if matrix is already sparse
  if not isinstance(co_matrix, pd.SparseDtype):
    # Convert dense matrix to sparse csr_matrix format
    co_matrix_sparse = csr_matrix(co_matrix.values, dtype=float)
  else:
    # Use the existing sparse matrix
    co_matrix_sparse = co_matrix

  # Calculate normalized Laplacian matrix
  degree_matrix = np.diag(co_matrix_sparse.sum(axis=0))
  laplacian_matrix = degree_matrix - co_matrix_sparse

  # Get the second smallest eigenvector (Fiedler vector)
  _, eigenvectors = eigsh(laplacian_matrix, k=2, which='LM',  tol=1E-2)
  fiedler_vector = eigenvectors[:, 1]

  # Sort indices based on Fiedler vector values
  sorted_indices = fiedler_vector.argsort()

  # Reorder rows and columns based on sorted indices
  reordered_matrix = co_matrix.iloc[sorted_indices, :]
  reordered_matrix = reordered_matrix.iloc[:, sorted_indices]

  return reordered_matrix



# ---- Supporting Functions for Data Processing ----
def extract_numbers(text):
  """
  This function extracts all numbers from a string representation of a list.

  Args:
      text: The string containing the list representation (e.g., "[584,12839,129239,222]").

  Returns:
      A list of integers extracted from the string.
  """
  # Remove square brackets using slicing
  text_without_brackets = text[1:-1]
  # Use ast.literal_eval for safe conversion (handles potential malformed strings)
  try:
    number_list = ast.literal_eval(text_without_brackets)
  except (ValueError, SyntaxError):
    # Handle potential exceptions during conversion (e.g., malformed string)
    raise ValueError('The string cannot be converted into a number list.')

  # Ensure all elements are integers
  if isinstance(number_list, int):
    return [number_list]
  else:
    return [int(num) for num in number_list]  # List comprehension for conversion

# Used in smartRPA-2-ActionLogger
def get_indexes_for_identifiers(motif_spots, identifiers):
    """
    Retrieves index values for each unique identifier in a list of identifiers based on a list of motif spots.

    Parameters:
    motif_spots (list): List of integer index values from the dataframe.
    identifiers (list): List of identifiers corresponding to the index positions.

    Returns:
    dict: A dictionary where keys are unique identifiers and values are lists of index values.
    """
    identifier_index_map = {}
    #if len(motif_spots) != len(identifiers):
    #    raise ValueError("motif_spots and identifiers must be of the same length.")

    for idx, identifier in enumerate(identifiers):
        if identifier not in identifier_index_map:
            identifier_index_map[identifier] = []
        identifier_index_map[identifier].append(motif_spots[idx])
        #print(identifier_index_map)

    return identifier_index_map

def compare_sets(set1, set2, n):
  """
  This function compares two sets of numbers represented as strings and identifies values within a range.

  Args:
      set1: The first originally inserted motifs in the validation log
      set2: The discovered motifs from the stumpy algorithm
      n: The range, e.g., half the window size, to discover missalignment

  Returns:
      identified_values: The motifs in the original validation log indexes of matches
      motif_values: The motifs index based on the motif stumpy discovery
      set_matches: A dataframe containing the values and the alignment score
  """
  set_matches = pd.DataFrame(columns=["originalMotif","discoveredMotif","alignmentAccuracy"])
  identified_values = []
  motif_values = []
  for num1 in set1:
    for num2 in set2:
      # Check if values are within the range n (considering absolute difference)
      if abs(num1 - num2) <= n:
        identified_values.append(num1)
        motif_values.append(num2)
        dict1 = {"originalMotif": num1, "discoveredMotif": num2, "alignmentAccuracy": abs(num1 - num2)}
        set_matches = set_matches._append(dict1, ignore_index=True)
        break  # Avoid duplicates if multiple values in set2 are within range

  return identified_values, motif_values, set_matches


def read_data_for_processing(isSmartRPA2024: bool = False,
                             isSmartRPA2025: bool = False,
                             isRealWorldTest: bool = False,
                             isActionLogger: bool = False,
                             leno_file_name: str = "202511_SR_RT_plus_extended.csv",
                             leno_gt_file_name: str = "202511_SR_RT_plus_extended_ground_truth.csv",
                             isHCI: bool = False,
                             log_name_smartRPA: str = "LenLog_1_1_10_25_5_5000.csv",
                             folder_path_smartRPA2025: str = "../logs/smartRPA/202511-update/"):
  # Create Standard Based data for both logs, so that all down stream measures look the same

  # 1. Dataframe mit Start Index, Length, End_Index
  # 2. Context Parameters
  #   2.2. All levels initiated
  #   2.1. Flatten hierarchy_columns
  # 3. Read Files

  # ground_truth => Dataframe with columns: caseid, start_index, length, end_index
  # log => Dataframe with all events symbolized according to hierarchy_columns
  # hierarchy_columns => List of Lists with all hierarchy levels used for symbolization
  # hierarchy_list => Flattened list of hierarchy_columns

  if isSmartRPA2025 or isSmartRPA2024 or isRealWorldTest:
    # Note: Read the alignment hints above
    systems = ["category"]
    applications = ["application"]
    uiGroup1 = ["workbook","browser_url","title"]
    uiGroup2 = ["current_worksheet"] 
    uiGroup3 = ["cell_range","cell_range_number"]
    uiElement = ["mouse_coord","tag_category","xpath","event_src_path","event_dest_path","tag_name","tag_title","tag_href",
                "tag_innerText","tab_pinned","tab_audible","tab_muted","window_ingognito","tab_moved_from_index","tab_moved_to_index","id"]
    actions = ["concept:name","eventQual"]
    # ---- Add all hierarchy levels into a list ----
    hierarchy_list_smartrpa = [systems, applications, uiGroup1, uiGroup2, uiGroup3, uiElement, actions]
    hierarchy_columns_smartrpa = list(chain.from_iterable([h for h in hierarchy_list_smartrpa if h]))
    
    hierarchy_list = hierarchy_list_smartrpa
    hierarchy_columns = hierarchy_columns_smartrpa
    hierarchy_columns_app_switch = list(chain.from_iterable([systems,applications]))

    sep_smartrpa = ","
    if isSmartRPA2024:
      # ---- Define folder path and relevant files ----
      folder_path_smartRPA = "../logs/smartRPA/percentageComparison/"
      percentagData_filename = "validationDataPercentage.csv"
      # ---- Getting the relevant files from the folder and sorting them into different lists for processing ----
      percentageLogs = []

      for file_in_folder in os.listdir(folder_path_smartRPA):
          if file_in_folder.startswith("LenLog"):
              percentageLogs.append(file_in_folder)

      # Check if data created for percentage based comparison is available
      try:
          percentageValData = pd.read_csv(folder_path_smartRPA + percentagData_filename)
          percentageValAvailable = True
      except FileNotFoundError as e:
          print(f"Could not read {percentagData_filename} from the folder.\n{e}")
          percentageValAvailable = False
      
      # ---- Read the relevant log file ----
      file = pd.read_csv(folder_path_smartRPA + log_name_smartRPA, sep=sep_smartrpa)
      print(f"Processing file: {log_name_smartRPA} with {len(file)} events.")
      comparisonVariables = percentageValData.loc[percentageValData['uiLogName'] == log_name_smartRPA]
      insertSpots = comparisonVariables["motifSpots"]
      parsed_spots = insertSpots.apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
      insert_spots_flat = [int(i) for sublist in parsed_spots for i in sublist]
      insert_spots_set = set(insert_spots_flat)
      print(f"{len(insert_spots_set)} insertion spots.")
      columns = ["caseid", "start_index", "length", "end_index"]
      ground_truth = pd.DataFrame(columns=columns)
      i = 0
      for insert_spot in insert_spots_set:
          ground_truth.loc[len(ground_truth)] = [i, insert_spot, comparisonVariables["motifLength"].iloc[0], insert_spot + comparisonVariables["motifLength"].iloc[0]]
          i += 1
          
    elif isSmartRPA2025:
      # ---- Define folder path and relevant files ----
      grundTruthOverallData = "validationLogInformation.csv"
      # ---- Getting the relevant files from the folder and sorting them into different lists for processing ----
      percentageLogs = []
      sep_smartrpa = ','
      file = pd.read_csv(folder_path_smartRPA2025 + log_name_smartRPA, sep=sep_smartrpa)
      print(f"Processing file: {log_name_smartRPA} with {len(file)} events.")

      groundTruthDataDF = pd.read_csv(folder_path_smartRPA2025 + grundTruthOverallData, sep=sep_smartrpa)

      groundTruthDataForLog = groundTruthDataDF[groundTruthDataDF['uiLogName'] == log_name_smartRPA]
      ground_truth = pd.DataFrame(columns=["caseid", "start_index", "length", "end_index"])
      motif = 0
      while motif < groundTruthDataForLog["noOfMotifs"].values[0]:
          # Get the regular parameters for the motif
          caseid = groundTruthDataForLog[f'motif{motif}-caseID'].values[0]
          motif_length = groundTruthDataForLog[f'motif{motif}-length'].values[0]
          occurances = groundTruthDataForLog[f'motif{motif}-occurances'].values[0]
          shuffle = groundTruthDataForLog[f'motif{motif}-shuffle'].values[0]
          # Get the start indexes for the motif
          start_index = groundTruthDataForLog[f'motif{motif}-startIndexes']
          parsed_spots = start_index.apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
          insert_spots_flat = [int(i) for sublist in parsed_spots for i in sublist]
          insert_spots_set = set(insert_spots_flat)
          # Iterate over the start indexes and create new rows in the ground truth dataframe
          for element in insert_spots_set:
              new_row = {"motif_number": motif,
                        "caseid": caseid,
                        "start_index": element,
                        "length": motif_length,
                        "end_index": int(element + motif_length - 1),
                        "total_occurances": int(occurances),
                        "shuffle": int(shuffle)}
              ground_truth = pd.concat([ground_truth, pd.DataFrame([new_row])], ignore_index=True)
          motif += 1

    elif isRealWorldTest:
        folder_path_realworld = "../logs/Banking/"
        realworld_log_name = "RunningExampleSmartRPA_Anonymized.csv"
        file = pd.read_csv(folder_path_realworld + realworld_log_name, sep=";")

        print(f"Processing file: {realworld_log_name} with {len(file)} events.")

        ground_truth = pd.DataFrame(columns=["caseid", "start_index", "length", "end_index"])
        # Manually define ground truth based on known inserted motifs
        ground_truth.loc[len(ground_truth)] = [0, 291, 55, 346]
        ground_truth.loc[len(ground_truth)] = [1, 1229, 52, 1281]
        ground_truth.loc[len(ground_truth)] = [2, 1508, 36, 1544]
        ground_truth.loc[len(ground_truth)] = [3, 2330, 35, 2365]
        ground_truth.loc[len(ground_truth)] = [4, 2662, 33, 2695]
        ground_truth.loc[len(ground_truth)] = [5, 2712, 40, 2752]
        ground_truth.loc[len(ground_truth)] = [6, 2764, 37, 2801]
        ground_truth.loc[len(ground_truth)] = [7, 2923, 45, 2968]
  elif isActionLogger:
      folder_path_leno = "../logs/leno/"
      sep_leno = ";"

      systems = []
      applications = ["targetApp"]
      uiGroup1 = ["url","target.workbookName"]
      uiGroup2 = ["target.sheetName"]
      uiElement = ["target.id","target.tagName","target.type","target.name","target.href"]
      actions = ["eventType"]
      hierarchy_list_leno = [applications, uiGroup1, uiGroup2, uiElement, actions]
      # Flatten the hierarchy list and remove empty lists
      hierarchy_list = hierarchy_list_leno
      hierarchy_columns = list(chain.from_iterable([h for h in hierarchy_list_leno if h]))
      hierarchy_columns_app_switch = list(chain.from_iterable([systems,applications]))
      #### Read the relevant log file ####
      # Choose between Leno Log Sequential or Parallel
      file = pd.read_csv(folder_path_leno + leno_file_name, sep=sep_leno)
      ground_truth_file = pd.read_csv(folder_path_leno + leno_gt_file_name, sep=sep_leno)
      insert_spots_leno = ground_truth_file["start_index"].astype(int).tolist()
      ground_truth_file["end_index"] = 0
      ground_truth_file["end_index"] = ground_truth_file["start_index"] + ground_truth_file["length"]
      ground_truth_file["total_occurances"] = 50
      ground_truth = ground_truth_file
      print(f"Processing file: {leno_file_name} with {len(file)} events.")

  elif isHCI:
      systems = ["machine"]
      uiElement = ["idUIelement"]
      actions = ["idUIaction"]
      hierarchy_list_hci = [systems, uiElement, actions]
      hierarchy_columns_hci = list(chain.from_iterable([h for h in hierarchy_list_hci if h]))
      hierarchy_list = hierarchy_list_hci
      hierarchy_columns = hierarchy_columns_hci
      hierarchy_columns_app_switch = list(chain.from_iterable([systems]))

      folder_path_HCI = "../logs/datasetForSmarterUX/"
      hci_log_name = "raw_interactions.csv"
      hci_ground_truth_name = "sequences_df_prep_EN.csv"

      file = hci_log = pd.read_csv(folder_path_HCI + hci_log_name, sep=";")
      hci_ground_truth = pd.read_csv(folder_path_HCI + hci_ground_truth_name, sep=",")
      # Split the idUIelement into element and action
      hci_log[["idUIelement", "idUIaction"]] = hci_log["idUIelement"].str.extract(r'([A-Za-z]+[0-9]+)(.*)', expand=True)
      ground_truth = pd.DataFrame(columns=["motif_id", "start_index", "length", "end_index", "discovery_count"])
      print(f"Processing file: HCI Dataset for Smarter UX with {len(hci_log)} events.")
      for idx, row in hci_ground_truth.iterrows():
          start_index = hci_log.index[hci_log["epoch"] == row["initepoch"]][0]
          end_index = hci_log.index[hci_log["epoch"] == row["endepoch"]][0]
          length = end_index - start_index +1
          discovery_count = 1
          ground_truth.loc[len(ground_truth)] = [idx, start_index, length, end_index, discovery_count]
      ground_truth.insert(0, 'caseid', range(0, 0 + len(ground_truth)))
  else:
      raise ValueError("No valid log selected. Please select one of the available options to TRUE.")


  log = grammar_util.symbolize_UILog(file, hierarchy_columns)

  ground_truth_start_list = list(zip(ground_truth["caseid"], ground_truth["start_index"], ground_truth["length"]))

  return {
      "hierarchy_list": hierarchy_list,
      "hierarchy_columns": hierarchy_columns,
      "hierarchy_columns_app_switch": hierarchy_columns_app_switch,
      "file": file,
      "log": log,
      "ground_truth": ground_truth,
      "ground_truth_start_list": ground_truth_start_list
   }


def shuffle_x_percent(df: pd.DataFrame, percentage: float, seed=None) -> pd.DataFrame:
    """
    Move floor(n * percentage) rows to random new positions inside the dataframe,
    while preserving the original order of all remaining rows.
    """

    if seed is not None:
        np.random.seed(seed)

    n = len(df)
    k = int(np.floor(n * percentage))  # number of rows to move
    if k <= 0:
        return df.copy()

    # Pick k rows to relocate
    indices_to_move = np.random.choice(n, size=k, replace=False)

    # Keep order in the original df for the rows NOT moved
    df_fixed = df.drop(index=indices_to_move).reset_index(drop=True)

    # Extract rows to move, but keep their original order
    df_move = df.iloc[sorted(indices_to_move)].reset_index(drop=True)

    # Generate k random insertion positions into df_fixed
    insert_positions = np.random.choice(len(df_fixed) + 1, size=k, replace=False)
    insert_positions.sort()  # insert from low to high to avoid index shifts

    # Insert each row at the chosen position
    result = df_fixed.copy()
    offset = 0
    for pos, (_, row) in zip(insert_positions, df_move.iterrows()):
        # Insert row
        before = result.iloc[:pos+offset]
        after  = result.iloc[pos+offset:]
        result = pd.concat([before, pd.DataFrame([row]), after], ignore_index=True)
        offset += 1

    return result

def generate_log_from_data_seed(dfAll: pd.DataFrame, 
                                randomDF: pd.DataFrame,
                                validation_path: str, 
                                concept_name_column: str, 
                                only_case_actions_df: pd.DataFrame,
                                random_cases_list: pd.DataFrame,
                                motifs: int,
                                occurances: int,
                                length: int,
                                percentageMotif: float,
                                shuffle: float):

  # ---- Generating the validation data ----

  log_name = f"log_motifs{motifs}_occurances{occurances}_length{length}_percentage{percentageMotif}_shuffle{shuffle}.csv"
  new_row = {
      "uiLogName": log_name,
      "percentageMotifsOverLog": percentageMotif,
      "noOfMotifs": motifs
  }

  # Calculate log length for percentage of motif to log
  log_total_length = motifs * length * occurances * 100 / percentageMotif
  log_noise_length = log_total_length * (100-percentageMotif)/100
  print(f"Processing: {log_name} with the total length {log_total_length} containing {log_noise_length} events.")

  if percentageMotif == 100:
    print("100% motifs requested — generating motif-only log without noise.")

    # noise log is empty
    random_log = pd.DataFrame(columns=dfAll.columns)

    # total number of motif insertions
    total_inserts = motifs * occurances

    # generate deterministic insertion positions
    insertion_indexes = list(range(0, total_inserts * length, length))

    # build indexes_df accordingly
    indexes_df = pd.DataFrame({
        "random_index": insertion_indexes,
        "uplift": 0,
        "final_index": insertion_indexes,
        "isUsed": False,
        "insertedCaseID": "",
        "end_indexes": [idx + length - 1 for idx in insertion_indexes]
    })

    # create shuffled case-ID sequence exactly as before
    case_sequence = (
        random_cases_list["case:concept:name"]
        .repeat(occurances)
        .sample(frac=1)
        .reset_index(drop=True)
    )

    assigned_positions = list(zip(insertion_indexes, case_sequence))
    assigned_positions.sort(key=lambda x: x[0], reverse=True)

    # skip noise creation and jump to insertion loop
    goto_insertion_phase = True
  else:
      goto_insertion_phase = False

  new_row["logLength"] = int(log_total_length)
  # --------------------------------------------------------------
  # INSERTION PHASE (same code, but supports motif-only mode)
  # --------------------------------------------------------------

  if goto_insertion_phase:
      # (random_log already empty; indexes_df already defined)
      pass
  else:
    # Generate random log just for noise
    random_log = randomDF[:int(log_noise_length)]
    random_log = random_log.drop(columns=["Unnamed: 0"],  errors='ignore')

    # Get random indexes for motif insertion
    insertion_indexes = sorted(np.random.choice(range(0, len(random_log)), size=motifs*occurances, replace=False),reverse=True)
    indexes_df = pd.DataFrame(columns=["random_index","uplift","final_index","isUsed"])
    indexes_df["random_index"] = insertion_indexes
    # How much will the element be uplifted due to prior insertions
    indexes_df["uplift"] = (len(indexes_df) - indexes_df.index-1) * length
    # Final indexes after uplift
    indexes_df["final_index"] = indexes_df["random_index"] + indexes_df["uplift"]
    # Mark all as not used << Not needed, but for checking
    indexes_df["isUsed"] = False
    # Case-ID sequence for insertion
    indexes_df["insertedCaseID"] = ""
    # For checking purposes, store the end index of the motif
    indexes_df["end_indexes"] = indexes_df["final_index"] + length - 1

    # Create the case-ID sequence for insertion in random order
    case_sequence = (
        random_cases_list["case:concept:name"]
        .repeat(occurances)
        .sample(frac=1)                 # << shuffle the repeated sequence
        .reset_index(drop=True)
    )

    # Safety check
    if len(insertion_indexes) < len(case_sequence):
        raise ValueError("Not enough insertion positions for all motif occurrences.")

    # Assign case-IDs to insertion positions (one-to-one)
    assigned_positions = list(zip(insertion_indexes[:len(case_sequence)], case_sequence))

    # Process from highest index to lowest
    assigned_positions.sort(key=lambda x: x[0], reverse=True)

  for position, case_id in assigned_positions:
      # Get the motif actions for the current case_id
      motif_actions = only_case_actions_df[only_case_actions_df[concept_name_column] == case_id][:length]
      # Shuffle the motif by the specified percentage
      motif_actions = shuffle_x_percent(motif_actions.reset_index(), shuffle, seed=42)
      # Insert motif actions into the random log at the specified position
      top_part = random_log.iloc[:position]
      bottom_part = random_log.iloc[position:]
      random_log = pd.concat([top_part, motif_actions, bottom_part]).reset_index(drop=True)
      # Mark the position as used
      indexes_df.loc[indexes_df["random_index"] == position, "isUsed"] = True
      # Store insertion details in indexes_df
      indexes_df.loc[indexes_df["random_index"] == position, "insertedCaseID"] = case_id

  random_log.reset_index(drop=True, inplace=True)
  i=0
  while i < motifs:
      new_row[f"motif{i}-length"] = length
      new_row[f"motif{i}-occurances"] = occurances
      new_row[f"motif{i}-shuffle"] = shuffle
      case_id = random_cases_list["case:concept:name"].iloc[i]
      new_row[f"motif{i}-caseID"] = case_id
      start_indexes = indexes_df[indexes_df["insertedCaseID"] == case_id]["final_index"].tolist()
      new_row[f"motif{i}-startIndexes"] = start_indexes
      i += 1

  random_log.to_csv(os.path.join(validation_path, log_name), index=False)
  return new_row

def print_progress_bar(current, total, bar_length=40):
    """
    Prints a progress bar to stdout in Jupyter, overwriting the previous line.
    """
    fraction = current / total
    filled = int(bar_length * fraction)
    bar = "█" * filled + "-" * (bar_length - filled)
    now = datetime.now()
    print(f"\rProgress: |{bar}| {current}/{total} ({fraction*100:.1f}% @ {now.time()})\n", end="")


