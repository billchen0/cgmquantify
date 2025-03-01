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


def range_glu(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the range of glucose values (max-min) for each subject.

    Args:
        df (pd.DataFrame): DataFrame with 'id' and 'gl' columns (subject ID and glucose values).

    Returns:
        pd.DataFrame: A DataFrame with one row per subject.
    """
    if not {"id", "gl"}.issubset(df.columns):
        raise ValueError("DataFrame must contain 'id' and 'gl' columns.")

    # Group by 'id' and compute range for each subject

    result = (
        df.groupby("id")["gl"]
        .agg(
            lambda x: {f"range": np.nanmax(x)-np.nanmin(x)}
        )
        .apply(pd.Series)
        .reset_index()
    )

    return result
