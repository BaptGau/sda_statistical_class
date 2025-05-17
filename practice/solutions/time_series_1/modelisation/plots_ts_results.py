from matplotlib import pyplot as plt
from pandas import DataFrame

from demos.utlis import setup_plot


def plot_preds_results(
    data: DataFrame,
    statsmodels_preds: DataFrame,
    statsforecast_preds: DataFrame,
):
    colors = setup_plot()

    plt.figure(figsize=(12, 8))
    plt.plot(data.index, data.co2, label="Actual", color=colors[0])
    plt.plot(
        statsmodels_preds.index, statsmodels_preds, label="Statsmodels", color=colors[1]
    )

    for idx, column in enumerate(statsforecast_preds.columns):
        style = ["--", "-", "dashed"]
        plt.plot(
            statsforecast_preds.index,
            statsforecast_preds[column],
            label=f"Statsforecast_{column}",
            color=colors[2],
            ls=style[idx],
        )
    plt.legend()
    plt.grid(True)
    plt.title("Differentiated CO₂ Levels Over Time (Monthly) & Predictions")
    plt.show()
