import pandas as pd
import numpy as np

def sd_glu(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the standard deviation of glucose values for each subject.

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
            lambda x: {f"SD": np.std(x)}
        )
        .apply(pd.Series)
        .reset_index()
    )

    return result
