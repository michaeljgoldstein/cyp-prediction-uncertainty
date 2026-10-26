import numpy as np
from src.cyp_uncertainty import soft_threshold_interval_error, risk_coverage_curve


def test_soft_threshold_interval_error():
    got = soft_threshold_interval_error(
        y_pred=[1.0, 2.0, 4.0],
        ci_lower=[0.0, 1.5, 1.0],
        ci_upper=[1.5, 2.5, 3.0],
    )
    np.testing.assert_allclose(got, [0.0, 0.0, 1.0])


def test_risk_coverage_curve():
    df = risk_coverage_curve(
        y_true=[0.0, 1.0, 2.0],
        y_pred=[0.1, 1.2, 3.0],
        uncertainty=[0.1, 0.2, 0.9],
        coverages=[1/3, 2/3, 1.0],
    )
    assert list(df.columns) == ["coverage", "n_retained", "mae"]
    assert len(df) == 3
    assert df.iloc[0]["mae"] <= df.iloc[-1]["mae"]
