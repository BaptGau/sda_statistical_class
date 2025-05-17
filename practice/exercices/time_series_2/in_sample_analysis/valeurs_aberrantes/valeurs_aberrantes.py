import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pandas import Series
from statsmodels.tsa.statespace.sarimax import SARIMAXResults, SARIMAX

from demos.utlis import setup_plot, get_time_series_with_outliers
from practice.solutions.time_series_2.in_sample_analysis.valeurs_aberrantes.valeurs_aberrantes import (
    plot_outliers,
)


def detect_outliers(residuals: Series) -> Series:
    """
    Detects outliers in a time series.

    Parameters:
    data (Series): The time series data.
    residuals (Series): The residuals of the model.

    Returns:
    Series: A boolean series indicating the outliers.
    """
    # Assuming residuals are gaussian, we can use the 3-sigma rule
    return residuals.abs() > 3 * residuals.std()


if __name__ == "__main__":
    data = get_time_series_with_outliers()
    colors = setup_plot()
    # todo: proposer un algorithme de détection des valeurs aberrantes (vu en cours, Isolation Forest, etc)

    model = SARIMAX(endog=data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))

    fitted_model = model.fit()

    in_sample_prediction = fitted_model.get_prediction(
        start=1, end=len(data)
    ).predicted_mean

    residuals = fitted_model.resid

    outliers = detect_outliers(residuals)

    plot_outliers(
        data=data,
        residuals=residuals,
        outliers=outliers,
        colors=colors,
        in_sample_prediction=in_sample_prediction,
    )

    plt.show()
