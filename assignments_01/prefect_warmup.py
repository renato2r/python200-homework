import numpy as np
import pandas as pd
from prefect import flow, task

# Input data from Q1
arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

@task
def create_series(arr):
    # Convert NumPy array to pandas Series named "values"
    return pd.Series(arr, name="values")

@task
def clean_data(series):
    # Remove NaN values from the Series
    return series.dropna()

@task
def summarize_data(series):
    # Return a dictionary with the requested statistical metrics
    # Using series.mode()[0] as requested in the hint
    summary = {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "mode": series.mode()[0]
    }
    return summary

@flow(name="Data Pipeline Flow")
def pipeline_flow(input_data):
    # Sequential execution of tasks
    series = create_series(input_data)
    cleaned = clean_data(series)
    summary = summarize_data(cleaned)
    
    # Print results to console
    for key, value in summary.items():
        print(f"{key.capitalize()}: {value:.2f}")
        
    return summary

if __name__ == "__main__":
    # Execute the flow
    pipeline_flow(arr)

"""
ANSWERS TO QUESTIONS:

1. This pipeline is simple -- just three small functions on a handful of numbers. Why might Prefect be more overhead than it is worth here?
For a script this small, Prefect adds complexity in terms of environment setup, 
database management (SQLite), and API connectivity. 

2. Describe some realistic scenarios where a framework like Prefect could still be useful, even if the pipeline logic itself stays simple like in this case.
- Retries: If 'create_series' pulled data from an unstable SQL ERP database, 
  Prefect could automatically retry the task if it fails due to network issues.
- Scheduling: If this summary needs to run every morning at 8:00 AM without 
  manual intervention.
- Monitoring & Alerts: If the pipeline fails, Prefect can send a notification 
  (Slack/Email) so the developer knows immediately without checking logs.
- Observability: When multiple people need to see if a process succeeded 
  through a UI (Dashboard) instead of reading raw terminal output.
"""