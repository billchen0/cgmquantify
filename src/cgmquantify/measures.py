import pandas as pd
import numpy as np


def cv_glu(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the coefficient of variation (SD/mean) for each subject.

    Args:
        df (pd.DataFrame): DataFrame with 'id' and 'gl' columns (subject ID and glucose values).

    Returns:
        pd.DataFrame: A DataFrame with one row per subject, each threshold as a separate column.
    """
    if not {"id", "gl"}.issubset(df.columns):
        raise ValueError("DataFrame must contain 'id' and 'gl' columns.")

    # Compute the percentage of values above each threshold for each subject
    result = (
        df.groupby("id")["gl"]
        .agg(
            lambda x: {f"CV": (np.std(x)/np.nanmean(x))*100}
        )
        .apply(pd.Series)
        .reset_index()
    )

    return result  # Correctly formatted wide-format DataFrame
