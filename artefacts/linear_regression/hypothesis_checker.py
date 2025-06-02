import numpy as np
import statsmodels.api as sm

from numpy._typing import ArrayLike
from scipy.stats import shapiro
from statsmodels.regression.linear_model import RegressionResults
from statsmodels.stats._lilliefors import lilliefors
from statsmodels.stats.diagnostic import het_goldfeldquandt
from statsmodels.stats.stattools import durbin_watson


def fit_ols(X: ArrayLike, y: ArrayLike) -> RegressionResults:
    """
    Fit an Ordinary Least Squares (OLS) regression model.

    Parameters:
        X (ArrayLike): Feature matrix.
        y (ArrayLike): Target vector.

    Returns:
        RegressionResults: Fitted OLS model.
    """
    X_const = sm.add_constant(X)
    return sm.OLS(y, X_const).fit()


def check_linearity(model: RegressionResults) -> bool:
    """
    Check linearity assumption based on R-squared.

    Parameters:
        model (RegressionResults): Fitted regression model.

    Returns:
        bool: True if R-squared > 0, indicating some linear relationship.
    """
    return model.rsquared > 0


def check_residuals_homoscedasticity(residuals: ArrayLike, X: ArrayLike) -> bool:
    """
    Test for homoscedasticity using the Goldfeld-Quandt test.

    Parameters:
        residuals (ArrayLike): Model residuals.
        X (ArrayLike): Feature matrix.

    Returns:
        bool: True if residuals have constant variance (p > 0.05).
    """
    _, p_value, _ = het_goldfeldquandt(residuals, X)
    return p_value > 0.05


def check_residuals_normality(residuals: ArrayLike, n_samples: int) -> bool:
    """
    Test for normality of residuals using Shapiro-Wilk or Lilliefors test.

    Parameters:
        residuals (ArrayLike): Model residuals.
        n_samples (int): Number of observations.

    Returns:
        bool: True if residuals are normally distributed (p > 0.5).
    """
    if n_samples < 1000:
        _, p_value = shapiro(residuals)
    else:
        _, p_value = lilliefors(residuals)
    return p_value > 0.5


def check_residuals_autocorrelation(residuals: ArrayLike) -> bool:
    """
    Test for autocorrelation in residuals using the Durbin-Watson statistic.

    Parameters:
        residuals (ArrayLike): Model residuals.

    Returns:
        bool: True if Durbin-Watson statistic is between 1.5 and 2.5.
    """
    dw_stat = durbin_watson(residuals)
    return 1.5 < dw_stat < 2.5


def check_no_colinearity(X: ArrayLike) -> bool:
    """
    Check for multicollinearity among features using correlation threshold.

    Parameters:
        X (ArrayLike): Feature matrix.

    Returns:
        bool: True if no feature pair has absolute correlation >= 0.8.
    """
    correlation_matrix = np.corrcoef(X, rowvar=False)
    if correlation_matrix.ndim == 0:
        return True

    coef_to_check = correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]
    return np.all(np.abs(coef_to_check) < 0.8)


def check_all_hypotheses(X: ArrayLike, y: ArrayLike, model: RegressionResults) -> dict:
    """
    Run a full diagnostic check of linear regression assumptions.

    Parameters:
        X (ArrayLike): Feature matrix.
        y (ArrayLike): Target vector.
        model (RegressionResults): Fitted regression model.

    Returns:
        dict: Dictionary of hypothesis check results.
    """
    residuals = model.resid
    n_samples = len(y)

    return {
        "linearity": check_linearity(model),
        "residuals_normality": check_residuals_normality(residuals, n_samples),
        "residuals_homoscedasticity": check_residuals_homoscedasticity(residuals, X),
        "residuals_no_autocorrelation": check_residuals_autocorrelation(residuals),
        "features_no_multicolinearity": check_no_colinearity(X),
    }


def format_check_report(hypothesis_results: dict) -> str:
    """
    Format the results of hypothesis checks into a readable report.

    Parameters:
        hypothesis_results (dict): Dictionary of boolean hypothesis results.

    Returns:
        str: Formatted report string.
    """
    return f"""--- Hypothesis check report ---
- Linearity: {hypothesis_results["linearity"]}
- Normality of the residuals: {hypothesis_results["residuals_normality"]}
- Homoscedasticity of the residuals: {hypothesis_results["residuals_homoscedasticity"]}
- No autocorrelation of the residuals: {hypothesis_results["residuals_no_autocorrelation"]}
- No multicolinearity in the features: {hypothesis_results["features_no_multicolinearity"]}
"""
