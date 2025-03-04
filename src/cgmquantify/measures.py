import pandas as pd
import numpy as np

def iqr_glu(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the distance between 25th and 75th percentile of glucose values for each subject.

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
            lambda x: {f"IQR": np.nanquantile(x,0.75)-np.nanquantile(x,0.25)}
        )
        .apply(pd.Series)
        .reset_index()
    )

    return result