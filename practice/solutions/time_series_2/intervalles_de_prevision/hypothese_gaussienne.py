import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from demos.utlis import (
    get_time_series,
    setup_plot,
    to_nixtla_format,
    from_nixtla_format,
)
from statsforecast.models import ARIMA as sf_arima
from statsforecast import StatsForecast


def plot_statsmodels_predictions_intervals(
    data: pd.DataFrame,
    statsmodels_preds: dict[str, pd.DataFrame],
    colors: list[str],
):
    plt.figure(figsize=(12, 8))
    plt.plot(data.index, data.co2, label="Actual", color=colors[0])

    pred_index = statsmodels_preds.get("point_forecast").index
    plt.plot(
        pred_index,
        statsmodels_preds.get("point_forecast"),
        label=f"Point forecasting",
        color=colors[2],
        ls="dashed",
    )

    plt.fill_between(
        pred_index,
        statsmodels_preds.get("low_95"),
        statsmodels_preds.get("high_95"),
        color=colors[3],
        alpha=0.2,
        label="95% PI",
    )
    plt.fill_between(
        pred_index,
        statsmodels_preds.get("low_80"),
        statsmodels_preds.get("high_80"),
        color=colors[3],
        alpha=0.8,
        label="80% PI",
    )

    plt.legend()
    plt.grid(True)
    plt.title(
        "Prévision ponctuelle et intervalles de confiance (Statsmodels)",
        fontweight="bold",
    )
    plt.show()


def plot_statsforecast_predictions_intervals(
    preds: pd.DataFrame,
    data: pd.DataFrame,
    colors: list[str],
    model_name: str = "ARIMA",
):
    plt.figure(figsize=(12, 8))
    plt.plot(data.index, data.co2, label="Actual", color=colors[0])

    pred_index = preds.index
    plt.plot(
        pred_index,
        preds.loc[:, model_name],
        label=f"Point forecasting",
        color=colors[2],
        ls="dashed",
    )

    plt.fill_between(
        pred_index,
        preds.loc[:, f"{model_name}-lo-95"],
        preds.loc[:, f"{model_name}-hi-95"],
        color=colors[3],
        alpha=0.2,
        label="95% PI",
    )
    plt.fill_between(
        pred_index,
        preds.loc[:, f"{model_name}-lo-80"],
        preds.loc[:, f"{model_name}-hi-80"],
        color=colors[3],
        alpha=0.8,
        label="80% PI",
    )

    plt.legend()
    plt.grid(True)
    plt.title(
        "Prévision ponctuelle et intervalles de confiance (Statsforecast)",
        fontweight="bold",
    )
    plt.show()


def statsmodels_modelisation(
    data: pd.DataFrame,
    target_name: str,
    steps: int = 12,
    p: int = 1,
    d: int = 1,
    q: int = 1,
) -> dict[str, pd.DataFrame]:
    ses_model = ARIMA(data.loc[:, target_name], order=(p, d, q)).fit()
    ses_preds = ses_model.get_forecast(steps=steps)

    prediction_intervals_95 = ses_preds.conf_int(alpha=0.05)

    low_95, high_95 = (
        prediction_intervals_95["lower co2"],
        prediction_intervals_95["upper co2"],
    )

    prediction_intervals_80 = ses_preds.conf_int(alpha=0.2)

    low_80, high_80 = (
        prediction_intervals_80["lower co2"],
        prediction_intervals_80["upper co2"],
    )

    return {
        "point_forecast": ses_preds.predicted_mean,
        "low_95": low_95,
        "high_95": high_95,
        "low_80": low_80,
        "high_80": high_80,
    }


def statsforecast_modelisation(
    data: pd.DataFrame,
    target_name: str,
    p: int = 1,
    d: int = 1,
    q: int = 1,
    steps: int = 12,
    level: list[int] | None = None,
) -> pd.DataFrame:
    models = [
        sf_arima(order=(p, d, q)),
    ]

    df = to_nixtla_format(data=data, target_name=target_name)

    sf = StatsForecast(models=models, n_jobs=-1, freq="W-SUN")  # hourly frequency

    sf.fit(df=df)

    if level:
        preds = sf.predict(h=steps, level=level)
    else:
        preds = sf.predict(h=steps)

    return from_nixtla_format(preds)


if __name__ == "__main__":

    data = get_time_series()
    colors = setup_plot()

    modelled_data = data.iloc[-500:, :]

    statsmodels_preds = statsmodels_modelisation(
        data=modelled_data, target_name="co2", p=4, q=2, d=1, steps=36
    )
    plot_statsmodels_predictions_intervals(
        data=modelled_data, statsmodels_preds=statsmodels_preds, colors=colors
    )

    statsforecast_preds = statsforecast_modelisation(
        data=modelled_data, target_name="co2", p=4, q=2, d=1, level=[80, 95], steps=36
    )

    plot_statsforecast_predictions_intervals(
        preds=statsforecast_preds,
        data=modelled_data,
        colors=colors,
    )
