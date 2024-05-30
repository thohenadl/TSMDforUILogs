import pandas as pd
import matplotlib.pyplot as plt

def plot_discovery_percentage_by_window_size_match(df):
  """
  This function creates a bar chart showing average DiscoveryPercentage for each windowSizeMatch value.

  Args:
      df (pd.DataFrame): The pandas dataframe containing the data.
  """

  # Group by windowSizeMatch and calculate average DiscoveryPercentage
  grouped_data = df.groupby('WindowSizeMatch')['DiscoveryPercentage'].mean().reset_index()

  # Extract data for plotting
  window_size_match = grouped_data['WindowSizeMatch'].tolist()
  avg_discovery_percentage = grouped_data['DiscoveryPercentage'].tolist()

  # Create the bar chart
  plt.figure(figsize=(10, 6))  # Adjust figure size as needed
  plt.bar(window_size_match, avg_discovery_percentage)
  plt.xlabel('WindowSizeMatch')
  plt.ylabel('Average Discovery Percentage')
  plt.title('Average Discovery Percentage by WindowSizeMatch')
  plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
  plt.grid(axis='y')
  plt.tight_layout()
  plt.show()


def plot_discovery_by_motif_length(df):
  """
  This function creates a box plot showing average DiscoveryPercentage for each MotifLength (considering only windowSizeMatch=False rows).

  Args:
      df (pd.DataFrame): The pandas dataframe containing the data.
  """

  # Filter data for windowSizeMatch=False
  filtered_df = df[df['WindowSizeMatch'] == 0.0]

  # Group by windowSizeMatch and calculate average DiscoveryPercentage
  grouped_data = filtered_df.groupby('motifLength')['DiscoveryPercentage'].mean().reset_index()

  # Extract data for plotting
  window_size_match = grouped_data['motifLength'].tolist()
  avg_discovery_percentage = grouped_data['DiscoveryPercentage'].tolist()

  # Create the bar chart
  plt.figure(figsize=(10, 6))  # Adjust figure size as needed
  plt.bar(window_size_match, avg_discovery_percentage)
  plt.xlabel('motifLength')
  plt.ylabel('Average Discovery Percentage')
  plt.title('Average Discovery Percentage by motifLength')
  plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
  plt.grid(axis='y')
  plt.tight_layout()
  plt.show()
