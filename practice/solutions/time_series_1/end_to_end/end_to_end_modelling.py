import pandas as pd
from matplotlib import pyplot as plt
from statsforecast.models import AutoETS, AutoARIMA, Holt, HoltWinters, AutoTBATS, ARIMA
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from artefacts.time_series.nixtla_plotter import plot_cross_validation_results
from demos.utlis import get_time_series, setup_plot, to_nixtla_format
from statsforecast import StatsForecast
from datasetsforecast.losses import rmse, mae, mape


def print_model_performance(model: str, y_true: pd.Series, y_pred: pd.Series):
    print(f" Model: {model} ".center(80, "="))
    print(f"RMSE: {rmse(y_true, y_pred):.2f}")
    print(f"MAE: {mae(y_true, y_pred):.2f}")
    print(f"MAPE: {mape(y_true, y_pred):.2f}%")


def cross_validation(data: pd.DataFrame, horizon: int):
    nixtla_data = to_nixtla_format(data=data, target_name="co2")

    models = [
        ARIMA(
            order=(3, 1, 1),
            season_length=12,
            seasonal_order=(1, 1, 1),
        ),
        AutoETS(season_length=12),
        AutoARIMA(season_length=12),
        Holt(),
        HoltWinters(season_length=12),
        AutoTBATS(season_length=12),
    ]

    sf = StatsForecast(models=models, freq="H", n_jobs=-1)

    crossvalidation_df = sf.cross_validation(
        df=nixtla_data, h=horizon, step_size=horizon, n_windows=5
    )

    crossvalidation_df.rename(
        columns={"y": "actual"}, inplace=True
    )  # rename actual values

    cutoff = crossvalidation_df["cutoff"].unique()

    plot_cross_validation_results(crossvalidation_df=crossvalidation_df, cutoff=cutoff)

    for model in ["AutoETS", "AutoARIMA", "Holt", "HoltWinters", "AutoTBATS"]:
        print_model_performance(
            "SARIMAX",
            crossvalidation_df.loc[:, "actual"],
            crossvalidation_df.loc[:, model],
        )


def validation(data: pd.DataFrame):
    train_data = data.iloc[:-horizon, :]
    test_data = data.iloc[-horizon:, :]

    sarimax = SARIMAX(
        train_data,
        order=(3, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit()
    sarimax_forecast = sarimax.forecast(steps=horizon)

    holt_winter = ExponentialSmoothing(
        train_data,
        trend="add",
        damped_trend=True,
        seasonal="add",
        seasonal_periods=12,
    ).fit(smoothing_trend=0.8, smoothing_seasonal=0.8)
    holt_winter_forecast = holt_winter.forecast(steps=horizon)

    plt.figure(figsize=(12, 6))
    plt.plot(train_data, label="train", color=colors[0])
    plt.plot(test_data, label="actual", color=colors[1])
    plt.plot(
        test_data.index,
        holt_winter_forecast,
        label="Holt-Winter",
        color=colors[2],
    )
    plt.plot(
        test_data.index,
        sarimax_forecast,
        label="SARIMAX",
        color=colors[3],
    )
    plt.legend()
    plt.grid(True)
    plt.show()

    print_model_performance(
        "Holt-Winter", test_data.values, holt_winter_forecast.values
    )
    print_model_performance("SARIMAX", test_data.values, sarimax_forecast.values)


if __name__ == "__main__":
    # Quel est le meilleur modèle ? Faites une sélection du modèles en comparant les erreurs de prédiction
    # horizon de 36 points
    horizon = 36
    data = get_time_series()
    colors = setup_plot()

    data = data.iloc[-500:, :]

    validation(data=data)
    cross_validation(data=data, horizon=horizon)
