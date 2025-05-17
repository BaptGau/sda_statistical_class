from statsforecast import StatsForecast
from statsforecast.models import ARIMA, Holt
from statsforecast.utils import AirPassengersDF
import matplotlib.pyplot as plt

from demos.utlis import setup_plot

if __name__ == "__main__":

    colors = setup_plot()

    df = AirPassengersDF
    df = df[df.unique_id == 1]

    sf = StatsForecast(
        models=[
            ARIMA(order=(3, 1, 1), seasonal_order=(1, 1, 1), season_length=12),
            Holt(season_length=12),
        ],
        freq="ME",
    )

    sf.fit(df)
    preds = sf.predict(h=12)
    print(preds.info())
    print(preds.head())

    # plt.figure(figsize=(12, 8))
    # plt.plot(df.ds, df.y, label="Actual", color=colors[0])
    # plt.plot(preds.ds, preds.ARIMA, label="Predicted", color=colors[1])
    #
    # plt.legend()
    # plt.grid()
    # plt.title("AirPassengers ARIMA forecast")
    # plt.show()
