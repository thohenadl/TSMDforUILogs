import pandas as pd
import os
import random
import numpy as np


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

def get_rand_uiLog(df, n_max=10, actions=9600):
    """Selects n random consecutive rows from a DataFrame.

    Args:
      df (pd.DataFrame): The DataFrame to select from.
      n_max (int): The upper limit for the random number function
      actions (int): Number of actions to be added into the UI log
          Default 9600 (8 hours * 60 minutes * 20 events/minute)

    Returns:
      pd.DataFrame: A DataFrame containing the selected rows.
    """
     # Use random sample and size parameter for efficiency
    ui_log = df.sample(min(len(df), actions), replace=True, ignore_index=True)
    
    # Ensure desired number of actions are present, adjusting n_max if necessary
    while len(ui_log) < actions and n_max < len(df):
        n_max += 1  # Increase n_max if not enough rows obtained
        additional_rows = df.sample(min(len(df) - len(ui_log), n_max), replace=True)
        ui_log = pd.concat([ui_log, additional_rows], ignore_index=True)
    
    return ui_log[:actions]  # Return only the desired number of actions

def get_random_values(df, column_name, m, min_len=1):
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
    
    if not 0 < reorder_percentage <= 1:
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
    # Get 
    # Ensure valid range for random numbers (from 0 to df_length - 1)
    valid_range = (0, len(df)-1)
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