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

def mad_glu(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the median absolute deviation (MAD) of glucose values for each subject.

    Args:
        df (pd.DataFrame): DataFrame with 'id' and 'gl' columns (subject ID and glucose values).

    Returns:
        pd.DataFrame: A DataFrame with one row per subject.
    """
    if not {"id", "gl"}.issubset(df.columns):
        raise ValueError("DataFrame must contain 'id' and 'gl' columns.")

    # Compute the median absolute deviation with scaling for each subject.
    result = (
        df.groupby("id")["gl"]
        .agg(
            lambda x: {f"MAD": stats.median_abs_deviation(x, scale = 'normal', nan_policy='omit')}
        )
        .apply(pd.Series)
        .reset_index()
    )

    return result

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

def quantile_glu(df: pd.DataFrame, quantiles=[0, 25, 50, 75, 100]) -> pd.DataFrame:
    """
    Compute the quantiles of glucose values.

    Args:
        df (pd.DataFrame): DataFrame with 'id' and 'gl' columns (subject ID and glucose values).
        quantiles (list): List of quantiles to calculate.

    Returns:
        pd.DataFrame: A DataFrame with one row per subject, each quantile as a separate column.
    """
    if not {"id", "gl"}.issubset(df.columns):
        raise ValueError("DataFrame must contain 'id' and 'gl' columns.")

    # Compute by groups - defined by id
    def compute_group_quantiles(group):
        return {f"X{t}": np.nanquantile(group["gl"], t/100) for t in quantiles}

    # Group by 'id' and compute quantiles for each group
    result = df.groupby("id").apply(compute_group_quantiles)

    # Convert results to DataFrame format
    result = result.apply(pd.Series).reset_index()

    return result  # Correctly formatted wide-format DataFrame
