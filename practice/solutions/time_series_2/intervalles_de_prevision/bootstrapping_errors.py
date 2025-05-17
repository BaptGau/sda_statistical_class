import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tools.typing import ArrayLike
from statsmodels.tsa.arima.model import ARIMA

from demos.utlis import get_time_series, setup_plot

import numpy as np
import pandas as pd
from typing import List
from numpy.typing import ArrayLike


def generate_prediction_intervals(
    forecast: pd.Series,
    residuals: ArrayLike,
    horizon: int = 12,
    n_bootstrap: int = 1000,
    level: List[float] = [0.8, 0.95],
) -> pd.DataFrame:
    # Initialize DataFrame with point forecast
    intervals_df = pd.DataFrame({"Point forecasting": forecast})

    # Array to store bootstrapped forecast results
    bootstrapped_results = np.zeros((horizon, n_bootstrap))

    # Generate bootstrapped forecast samples
    for bootstrapp in range(n_bootstrap):
        bootstrapped_errors = np.random.choice(residuals, size=horizon, replace=True)
        bootstrapped_results[:, bootstrapp] = forecast + bootstrapped_errors

    # Calculate prediction intervals for each confidence level
    for lev in level:
        lower_percentile = (1 - lev) / 2 * 100
        upper_percentile = (1 + lev) / 2 * 100
        lower_bound = np.percentile(bootstrapped_results, lower_percentile, axis=1)
        upper_bound = np.percentile(bootstrapped_results, upper_percentile, axis=1)

        # Store the results in the DataFrame
        intervals_df[f"low_{int(lev * 100)}"] = lower_bound
        intervals_df[f"high_{int(lev * 100)}"] = upper_bound

    return intervals_df


def plot_prediction_intervals(
    data: pd.DataFrame, intervals_df: pd.DataFrame, colors: list[str]
) -> None:
    plt.figure(figsize=(12, 8))
    plt.plot(data.index, data.co2, label="Actual", color=colors[0])

    pred_index = intervals_df.loc[:, "Point forecasting"].index
    plt.plot(
        pred_index,
        intervals_df.loc[:, "Point forecasting"],
        label=f"Point forecasting",
        color=colors[2],
        ls="dashed",
    )

    plt.fill_between(
        pred_index,
        intervals_df.loc[:, "low_95"],
        intervals_df.loc[:, "high_95"],
        color=colors[3],
        alpha=0.2,
        label="95% PI",
    )
    plt.fill_between(
        pred_index,
        intervals_df.loc[:, "low_80"],
        intervals_df.loc[:, "high_80"],
        color=colors[3],
        alpha=0.8,
        label="80% PI",
    )

    plt.legend()
    plt.grid(True)
    plt.title(
        "Prévision ponctuelle et intervalles de confiance (boostrapping errors)",
        fontweight="bold",
    )
    plt.show()


if __name__ == "__main__":
    data = get_time_series()
    colors = setup_plot()

    target_name = "co2"
    fitted_model = ARIMA(data.loc[:, target_name], order=(4, 1, 0)).fit()
    # plot model diagnostics
    fitted_model.plot_diagnostics()
    plt.show()
    # get residuals
    residuals = fitted_model.resid
    # get forecast
    forecast = fitted_model.forecast(steps=36)
    # generate prediction intervals
    intervals_df = generate_prediction_intervals(
        forecast=forecast, residuals=residuals, horizon=36
    )
    # plot intervals
    plot_prediction_intervals(data=data, intervals_df=intervals_df, colors=colors)
