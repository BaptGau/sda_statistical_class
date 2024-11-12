import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from demos.utlis import (
    get_time_series,
    setup_plot,
    to_nixtla_format,
    from_nixtla_format,
)
from statsforecast import StatsForecast
from statsforecast.models import Holt


if __name__ == "__main__":
    data = get_time_series()
    colors = setup_plot()
    # fit a trended exponential smoothing model and predict