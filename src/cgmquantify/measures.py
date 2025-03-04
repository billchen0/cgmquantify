import pandas as pd
from scipy import stats


def mad_glu(df: pd.DataFrame, scaling: float) -> pd.DataFrame:
    """
    Compute the median absolute deviation (MAD) of glucose values for each subject.

    Args:
        df (pd.DataFrame): DataFrame with 'id' and 'gl' columns (subject ID and glucose values).
        scaling (float): Scaling factor for MAD calculation.

    Returns:
        pd.DataFrame: A DataFrame with one row per subject.
    """
    if not {"id", "gl"}.issubset(df.columns):
        raise ValueError("DataFrame must contain 'id' and 'gl' columns.")

    # Compute the median absolute deviation with scaling for each subject.
    result = (
        df.groupby("id")["gl"]
        .agg(
            lambda x: {f"MAD": stats.median_abs_deviation(x, scale = scaling)}
        )
        .apply(pd.Series)
        .reset_index()
    )
    print(result)

    return result  # Correctly formatted wide-format DataFrame
