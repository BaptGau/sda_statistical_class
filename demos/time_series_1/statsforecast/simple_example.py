from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsforecast.utils import AirPassengersDF
import matplotlib.pyplot as plt

from demos.utlis import setup_plot

if __name__ == "__main__":

    colors = setup_plot()

    df = AirPassengersDF
    df = df[df.unique_id == 1].loc[:, ["ds", "y"]].set_index("ds")

    model = SARIMAX(
        endog=df["y"], order=(3, 1, 1), seasonal_order=(1, 1, 1, 12), freq="M"
    )

    fitted = model.fit()

    print(fitted.summary())
    preds = fitted.forecast(steps=12)

    print(preds.head())

    plt.figure(figsize=(12, 8))
    plt.plot(df.index, df.y, label="Actual", color=colors[0])
    plt.plot(preds.index, preds, label="Predicted", color=colors[1])
    plt.legend()
    plt.grid(True)
    plt.title("AirPassengers ARIMA forecast")
    plt.show()
