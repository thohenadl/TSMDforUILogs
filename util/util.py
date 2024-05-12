# If a function is added in here the Jupyter Notebook has to be restarted (Kernal restart) to load the function properly

import pandas as pd
import os
import random
import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix

import stumpy
import ast

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
    
def encoding_UiLog(uiLog: pd.DataFrame, cooccuranceBased: bool=False, cooccurance_distance: int=2):
    if cooccuranceBased:
        # Encode the application data by Cooccurance
        application_co_matrix = co_occurrence_matrix_n(uiLog, cooccurance_distance, "application")
        application_matrix = spectral_ordering_cooccurrence(application_co_matrix)
        applicationDict = createDict(list(application_matrix))
        # Encode the concept name (action) by Cooccurance
        concept_name_co_matrix = co_occurrence_matrix_n(uiLog, cooccurance_distance, "concept:name")
        concept_name_matrix = spectral_ordering_cooccurrence(concept_name_co_matrix)
        conceptNamesDict = createDict(list(concept_name_matrix))
    else:
        # Encode application by appearing order
        applicationDict = createDict(set(uiLog['application'].unique()))
        # Encode concept name (action) by appearing order sorted per application
        conceptNamesDict = createDict(list(uiLog.sort_values(by=['application'])['concept:name'].unique()))

    # Encode the categories with no special order as they are very few
    categoriesDict = createDict(set(uiLog.sort_values(by=['category'])['category'].unique()))
    # Add as new column to the dataframe
    uiLog['application:id'] = uiLog.apply(lambda row: get_key(row, applicationDict, 'application'), axis=1)
    uiLog['concept:name:id'] = uiLog.apply(lambda row: get_key(row, conceptNamesDict, 'concept:name'), axis=1)
    uiLog['category:id'] = uiLog.apply(lambda row: get_key(row, categoriesDict, 'category'), axis=1)

    # Encode all ids into a single value for univariate discovery by using tumples
    numbersDF = uiLog[['concept:name:id', 'application:id', 'category:id']]

    # Generate unique tuples for indexing the individual combinations of the rows mentioned
    unique_df = numbersDF.drop_duplicates(subset=numbersDF.columns, keep='first')
    tuples = [tuple(row[['concept:name:id', 'application:id', 'category:id']]) for i, row in unique_df.sort_values(by='application:id').iterrows()]
          
    uiLog['tuple:id'] = uiLog.apply(lambda row: get_id(row, tuples, columns=['concept:name:id','application:id', 'category:id']), axis=1)

    return uiLog

# ---- Motif Discovery ----

def discover_motifs(uiLog: pd.DataFrame, window_size: int=25):
    """
    Args:
      uiLog (DataFrame): Encoded uiLog containing the columns in text and integer format

    Returns:
      stumpy tm_matrix
    """
    starting_row = 0
    ending_row = len(uiLog)-1
    #Extract ids and rows
    ids = uiLog.loc[starting_row:ending_row,'tuple:id'].tolist()
    rows = [i for i in range(len(uiLog.loc[starting_row:ending_row,'tuple:id']))]

    event_series = uiLog.loc[starting_row:ending_row,'tuple:id'].values.astype(float)
    tm_matrix = stumpy.stump(event_series, window_size)

    return tm_matrix, event_series

def reduceLogToDiscovered(dataframe: pd.DataFrame, topMotifIndex: list, windowSize: int):
    new_df = pd.DataFrame(columns=dataframe.columns.tolist() + ["case"])  # Add case column
    case_id = 0
    for start_index in topMotifIndex:
      # Ensure start index is within dataframe bounds
      if start_index < 0 or start_index >= len(dataframe):
        continue
      end_index = min(start_index + windowSize, len(dataframe))  # Handle potential out-of-bounds end index
      window_df = dataframe.iloc[start_index:end_index].copy()
      window_df["case:concept:name"] = case_id
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

def get_rand_uiLog(df, n_max:int=10, actions:int=9600):
    """Selects random n consequitive rows from a DataFrame.

    Args:
      df (pd.DataFrame): The DataFrame to select from.
      n_max (int): The upper limit for the random number function
      actions (int): Number of actions to be added into the UI log
          Default 9600 (8 hours * 60 minutes * 20 events/minute)

    Returns:
      pd.DataFrame: A DataFrame containing the selected rows.
    """
    # Use random sample and size parameter for efficiency
    if n_max == 1:
      # For faster calculation
      return get_completely_random_uiLog(df,actions)
    
    ui_log = pd.DataFrame()
    while(len(ui_log) < actions):
        if (len(ui_log) % 1000) == 0:
          print(f"Current generated UiLog length: {len(ui_log)}")
        # Slow way, optimized by getting multiple random indecies at once
        index = random.randint(0,len(df)-n_max)
        sequence = df.iloc[index:index+n_max]
        concat_Series = [ui_log,sequence]
        ui_log = pd.concat(concat_Series)

    return ui_log

def get_completely_random_uiLog(df, actions=9600):
    if actions <= 0:
          raise ValueError("actions must be a positive integer")

    # Generate random starting indices efficiently using numpy.random.randint
    indices = random.sample(range(0, min(len(df)-1,actions)), min(len(df)-1,actions))
    while len(indices)-1 < actions:
        more_indices = random.sample(range(0, min(len(df)-1,actions)), min(len(df)-1,actions))
        indices = indices + more_indices

    # Select rows using efficient indexing and concatenation
    indices = indices[:actions]
    ui_log = pd.concat([df.iloc[i:i+1] for i in indices])

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
    return random_values.tolist()

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
    num_elements_to_reorder = int(len(df) * (reorder_percentage/100))
    
    # Randomly select elements to reorder, ensuring they are within valid range (0 to len(df) - 1)
    valid_range = (0, len(df) - 1)
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
    random.shuffle(random_cases_list)
    
    # Generate random numbers and limit them to the valid range
    # Make them descending to order without random overlap
    index_list = sorted([random.randint(0,len(uiLog)-1) for _ in range(len(random_cases_list))],reverse=True) 
    
    # Inserting the routines top down
    for i, routine in enumerate(random_cases_list):
        # Get the case elements and order by sorted_insert_col (timestamp)
        insert_df = dfcases[dfcases[case_column_name] == random_cases_list[i]].sort_values(sorted_insert_col)
        
        if reduced:
            insert_df = remove_n_percent_rows(insert_df,reduced_by)
        if shuffled:
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
    return []

  # Ensure all elements are integers
  return [int(num) for num in number_list]  # List comprehension for conversion

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