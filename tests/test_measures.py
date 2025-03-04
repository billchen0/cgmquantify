import pytest
from pathlib import Path
import pandas as pd
from src.cgmquantify import measures

@pytest.mark.parametrize(
    "function_name, output_name, kwargs",
    [
        ("quantile_glu", "X0", {"quantiles": [0]}),
        ("quantile_glu", "X25", {"quantiles": [25]}),
        ("quantile_glu", "X50", {"quantiles": [50]}),
        ("quantile_glu", "X75", {"quantiles": [75]}),
        ("quantile_glu", "X100", {"quantiles": [100]})
    ],
)
def test_cgm_measure_extraction(function_name, output_name, kwargs):
    """
    Generalized test for CGM measure extraction functions that return results as columns.

    Args:
        function_name (str): Name of the function in `measures.py`.
        output_name (str): The expect output column in `cgm_measures.csv` corresponding to this function.
        kwargs (dict): Additional arguments for the function.
    """
    # Load CGM data
    df = pd.read_csv("./data/cgm.csv")

    # Get function dynamically from measures module
    func = getattr(measures, function_name, None)
    assert callable(func), f"Function '{function_name}' not found in measures.py"

    # Compute CGM measures
    computed_result = func(df, **kwargs)

    # Load expected results
    #expected_df = pd.read_csv("tests/data/cgm_measures.csv")[["id", output_name]]
    expected_df = pd.read_csv("./data/cgm_measures.csv")[["id", output_name]]
    assert set(computed_result["id"]) == set(
        expected_df["id"]
    ), f"Subject ID mismatch in {function_name}"

    # Sort both dataframes by 'id' to ensure correct row-wise comparison
    computed_result = computed_result.sort_values("id").reset_index(drop=True)
    expected_df = expected_df.sort_values("id").reset_index(drop=True)

    # Extract only the output column for comparison
    computed_values = computed_result[output_name]
    expected_values = expected_df[output_name]

    # Compare values, allowing for floating point differences
    assert (
        computed_values.equals(expected_values)
        or (abs(computed_values - expected_values) < 1e-1).all()
    ), f"Mismatch in {output_name} for {function_name}"
