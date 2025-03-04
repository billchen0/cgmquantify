import pandas as pd
import numpy as np

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
