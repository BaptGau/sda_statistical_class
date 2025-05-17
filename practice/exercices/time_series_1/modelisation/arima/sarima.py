import warnings

import pandas as pd
import itertools

from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults

from demos.utlis import (
    get_time_series,
    setup_plot,
)


def choose_best_model(
    data: pd.Series,
    max_p=5,
    max_d=1,
    max_q=5,
    max_P=5,
    max_D=1,
    max_Q=5,
    max_s=12,
) -> SARIMAXResults:
    """
    Perform a grid search to find the best SARIMA model.

    Args:
        data (pd.Series): Time series data for the model.
        max_p (int): Autoregressive order.
        max_d (int): Differencing order.
        max_q (int): Moving average order.
        max_P (int): Seasonal autoregressive order.
        max_D (int): Seasonal differencing order.
        max_Q (int): Seasonal moving average order.
        max_s (int): Seasonal period.

    Returns:
        SARIMAXResults: Best SARIMA model based on AIC.
    """
    best_aic = float("inf")
    best_model = None
    best_params = None

    # Generate all parameter combinations
    pdq = list(itertools.product(range(max_p + 1), range(max_d + 1), range(max_q + 1)))
    seasonal_pdq = list(
        itertools.product(range(max_P + 1), range(max_D + 1), range(max_Q + 1), [max_s])
    )

    # Disable warnings during grid search
    warnings.filterwarnings("ignore")

    for seasonal_param in seasonal_pdq:
        for param in pdq:
            try:
                # Fit SARIMAX model
                model = SARIMAX(
                    data,
                    order=param,
                    seasonal_order=seasonal_param,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                results = model.fit(disp=False)

                # Check if this model is better (lower AIC)
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_model = results
                    best_params = (param, seasonal_param)

                    print(" New best model found! ".center(50, "="))
                    print(f"Parameters: {best_params}, AIC: {best_aic:.2f}")

            except Exception as e:
                # Handle cases where model fitting fails
                continue

    print(f"Best SARIMA parameters: {best_params}, AIC: {best_aic}")
    return best_model


if __name__ == "__main__":

    data = get_time_series()
    colors = setup_plot()

    modelled_data = data.iloc[-500:, :]  # We do not need to difference the data anymore

    choose_best_model(data=modelled_data)
    # todo : fit an SARIMA(X) model and predict
    # todo: la période de saisonnalité est de 12
