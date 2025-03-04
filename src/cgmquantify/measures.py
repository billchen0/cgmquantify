import pandas as pd
from scipy import stats


def mad_glu(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the mean absolute deviation (MAD) of glucose values for each subject.

    Args:
        df (pd.DataFrame): DataFrame with 'id' and 'gl' columns (subject ID and glucose values).

    Returns:
        pd.DataFrame: A DataFrame with one row per subject.
    """
    if not {"id", "gl"}.issubset(df.columns):
        raise ValueError("DataFrame must contain 'id' and 'gl' columns.")

    # Compute the percentage of values above each threshold for each subject
    result = (
        df.groupby("id")["gl"]
        .agg(
            lambda x: {f"MAD": stats.median_abs_deviation(x)}
        )
        .apply(pd.Series)
        .reset_index()
    )

    return result  # Correctly formatted wide-format DataFrame
