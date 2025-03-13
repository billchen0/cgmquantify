import pandas as pd
import numpy as np

def above_percent(df: pd.DataFrame, targets_above=[140, 180, 250]) -> pd.DataFrame:
    """
    Compute the percentage of glucose values above given thresholds for each subject.

    Args:
        df (pd.DataFrame): DataFrame with 'id' and 'gl' columns (subject ID and glucose values).
        targets_above (list): List of threshold values to compare glucose levels.

    Returns:
        pd.DataFrame: A DataFrame with one row per subject, each threshold as a separate column.
    """
    if not {"id", "gl"}.issubset(df.columns):
        raise ValueError("DataFrame must contain 'id' and 'gl' columns.")

    # Compute the percentage of values above each threshold for each subject
    result = (
        df.groupby("id")["gl"]
        .agg(
            lambda group: {
                f"above_{t}": (group > t).mean() * 100 for t in targets_above
            }
        )
        .apply(pd.Series)
        .reset_index()
    )

    return result  # Correctly formatted wide-format DataFrame

def mag_change_per_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the Mean Absolute Glucose change (MAG) as in the iglu package.

    Args:
        df (pd.DataFrame): DataFrame with columns 'id', 'time', and 'gl', where
                           'time' is a timestamp in the format 'YYYY-MM-DD HH:MM:SS'.
    
    Returns:
        pd.DataFrame: A DataFrame with columns 'id' and 'MAG'.
    """
    # Check for required columns
    if not {"id", "gl", "time"}.issubset(df.columns):
        raise ValueError("DataFrame must contain 'id', 'gl', and 'time' columns.")

    # Convert time column to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df["time"]):
        df["time"] = pd.to_datetime(df["time"])

    # Sort data by subject and time
    df = df.sort_values(by=["id", "time"])

    # Compute differences
    df["time_diff"] = df.groupby("id")["time"].diff().dt.total_seconds() / 60  # Time in minutes
    df["gl_diff"] = df.groupby("id")["gl"].diff().abs()

    # Compute the instantaneous rate, avoiding division by zero
    df["rate"] = df["gl_diff"] / df["time_diff"]
    df.loc[df["time_diff"] == 0, "rate"] = np.nan  # Prevent div-by-zero errors

    # Compute the mean rate and median time difference per subject
    agg_df = df.groupby("id", as_index=False).agg(
        mean_rate=("rate", "mean"),
        median_time_diff=("time_diff", "median")
    )

    # Compute MAG
    agg_df["MAG"] = agg_df["mean_rate"] * agg_df["median_time_diff"]

    return agg_df[["id", "MAG"]]



