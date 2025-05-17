import math

import matplotlib.pyplot as plt
import numpy as np
from pandas import Series
from sklearn.base import RegressorMixin
from sklearn.svm import SVR
from statsmodels.tsa.statespace.sarimax import SARIMAX

from demos.utlis import setup_plot, get_time_series_with_missing_values


def plot_serie(
    data: Series,
    null_indexes: Series,
    colors: list[str],
    null_value_label: str = "Valeurs manquantes",
):
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data, color=colors[0])
    # make a scatterplot to plot the missing values
    plt.scatter(
        x=null_indexes,
        y=data[null_indexes].apply(lambda x: 0 if math.isnan(x) else x),
        color="salmon",
        marker="o",
        label=null_value_label,
        s=20,
    )
    plt.legend()
    plt.grid(True)
    plt.title("Série temporelle avec valeurs manquantes")
    plt.show()


def robust_moving_average_interpolation(data: Series, window: int = 12) -> Series:
    """
    Interpolates missing values in a time series using a robust moving average.

    Parameters:
    data (Series): The time series data.
    window (int): The rolling window size.

    Returns:
    Series: The time series with missing values interpolated.
    """
    imputed = data.copy()
    robust_rolling_mean = data.rolling(window=window, min_periods=1).apply(
        lambda x: np.nanmean(x)
    )
    imputed[data.isnull()] = robust_rolling_mean
    return imputed


def polynomial_interpolation(data: Series, order: int) -> Series:
    """
    Interpolates missing values in a time series using polynomial interpolation.

    Parameters:
    data (Series): The time series data.
    order (int): The order of the polynomial to use.

    Returns:
    Series: The time series with missing values interpolated.
    """
    imputed = data.interpolate(method="polynomial", order=order)
    return imputed


def machine_learning_interpolation(
    data: Series, regressor: RegressorMixin, differentiate_serie: bool = False
) -> Series:
    """
    Interpolates missing values in a time series using a machine learning regressor.

    Parameters:
    data (Series): The time series data.
    regressor (RegressorMixin): The machine learning regressor to use.
    differentiate_serie (bool): Whether to differentiate the time series before fitting the model.

    Returns:
    Series: The time series with missing values interpolated.
    """
    data = data.copy()

    if differentiate_serie:
        original_data = data.copy()
        data = data.diff().dropna()

    missing_indices = data[data.isnull()].index
    known_indices = data[data.notnull()].index

    X_train = known_indices.values.reshape(-1, 1)
    y_train = data.loc[known_indices].values

    X_missing = missing_indices.values.reshape(-1, 1)

    regressor.fit(X_train, y_train)

    predicted_values = regressor.predict(X_missing)

    data.loc[missing_indices] = predicted_values

    if differentiate_serie:
        data = original_data.combine_first(data.cumsum())

    return data


def kalman_filter_interpolation(
    data: Series,
    p: int = 3,
    d: int = 1,
    q: int = 1,
    P: int = 1,
    D: int = 1,
    Q: int = 1,
    seasonal_frequency: int = 12,
) -> Series:
    """
    Interpolates missing values in a time series using a Kalman filter.

    Parameters:
    data (Series): The time series data.
    p (int): The AR order.
    d (int): The differencing order.
    q (int): The MA order.
    P (int): The seasonal AR order.
    D (int): The seasonal differencing order.
    Q (int): The seasonal MA order.
    seasonal_frequency (int): The frequency of the seasonality.

    Returns:
    Series: The time series with missing values interpolated.
    """

    imputed = data.copy()
    missing_indexes = data[data.isnull()].index

    model = SARIMAX(
        data,
        order=(p, d, q),
        seasonal_order=(P, D, Q, seasonal_frequency),
    ).fit()

    model_pred = model.predict(start=1, end=len(data))

    imputed.loc[missing_indexes] = model_pred.loc[missing_indexes]
    return imputed


if __name__ == "__main__":
    data, null_indexes = get_time_series_with_missing_values()
    colors = setup_plot()
    plot_serie(data=data, null_indexes=null_indexes, colors=colors)

    # moving average interpolation
    plot_serie(
        data=robust_moving_average_interpolation(data, window=12),
        null_indexes=null_indexes,
        colors=colors,
        null_value_label="Interpolation par moyenne mobile robuste",
    )

    # interpolation polynomiale
    order = 3
    plot_serie(
        data=polynomial_interpolation(data, order=order),
        null_indexes=null_indexes,
        colors=colors,
        null_value_label=f"Interpolation polynomiale (ordre {order})",
    )
    # ...

    # machine learning interpolation
    support_vector_machine = SVR()
    plot_serie(
        data=machine_learning_interpolation(
            data, regressor=support_vector_machine, differentiate_serie=False
        ),
        null_indexes=null_indexes,
        colors=colors,
        null_value_label="Interpolation par méthode de machine learning (Support Vector Machine)",
    )

    # kalman filter interpolation
    plot_serie(
        data=kalman_filter_interpolation(data),
        null_indexes=null_indexes,
        colors=colors,
        null_value_label="Interpolation par méthode de filtre de Kalman",
    )
    # todo: proposer un algorithme d'imputation des valeurs manquantes (polynome d'interpolation, moyenne mobile robuste, algorithme de ML, Filtre de Kalman par SARIMAX, etc.)
    # todo 2: comparer les performances des algorithmes d'imputation avec une méthode d'évaluation
