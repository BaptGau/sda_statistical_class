from enum import Enum
from functools import lru_cache
from typing import Optional, Tuple, Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from numpy._typing import ArrayLike
from sklearn.datasets import make_regression, make_classification
from statsmodels.discrete.discrete_model import BinaryResultsWrapper


class ProblemType(Enum):
    REGRESSION = "regression"
    CLASSIFICATION = "classification"


def setup_plot():
    import seaborn as sns

    sns.set_theme(style="whitegrid")

    return ["#004aad", "#2bb4d4", "#2e2e2e", "#5ce1e6"]


def get_data(type: ProblemType):
    if type == ProblemType.REGRESSION:
        return make_regression(n_samples=200, n_features=1, bias=4, noise=20.0)
    else:
        return make_classification(
            n_samples=200, n_features=2, n_informative=2, n_redundant=0, random_state=42
        )


def plot_data(
    X: ArrayLike, y: ArrayLike, colors: list[str], preds: Optional[ArrayLike] = None
):
    plt.figure(figsize=(12, 8))
    plt.scatter(X, y, color=colors[0], alpha=0.6, label="Données")
    plt.xlabel("X")
    plt.ylabel("y")
    if preds is not None:
        plt.plot(X, preds, color=colors[1], label="Prediction")
        plt.title("Prédiction du modèle")
    else:
        plt.title("Target vs features")

    plt.legend()
    plt.grid()
    plt.show()


def plot_classification(
    X: ArrayLike,
    y: ArrayLike,
    colors: list[str],
    clf: Optional[BinaryResultsWrapper] = None,
):
    cmap_background = ListedColormap([colors[0], colors[1]])
    cmap_points = ListedColormap([colors[0], colors[1]])

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

    if clf:
        # Flatten the meshgrid arrays and add a column of ones for the intercept
        grid = np.c_[xx.ravel(), yy.ravel()]
        grid_with_intercept = sm.add_constant(grid)

        # Predict and reshape to match the shape of the meshgrid
        Z = clf.predict(grid_with_intercept)
        Z = Z.reshape(xx.shape)

    plt.figure(figsize=(12, 8))
    if clf:
        plt.contourf(xx, yy, Z, alpha=0.2, cmap=cmap_background)

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_points, edgecolor=colors[2], s=50)
    if clf:
        plt.title(
            "Frontière de décision de la régression logistique", fontweight="bold"
        )
    else:
        plt.title("Visualisation du dataset", fontweight="bold")
    plt.xlabel("Variable 1")
    plt.ylabel("Variable 2")
    plt.grid(True)
    plt.show()


@lru_cache(maxsize=1)
def get_time_series():
    from statsmodels.datasets import co2

    data = co2.load_pandas().data
    # data.index.name = "date"
    weekly_data = data.resample("W").mean()
    return weekly_data.iloc[-500:, :]  # only keep the last 500 points


def to_nixtla_format(data: pd.DataFrame, target_name: str) -> pd.DataFrame:
    return pd.DataFrame(
        {"ds": data.index, "y": data.loc[:, target_name], "unique_id": target_name}
    )


def from_nixtla_format(data: pd.DataFrame) -> pd.DataFrame:
    return data.set_index("ds", drop=True)


def get_noisy_time_series() -> pd.Series:
    np.random.seed(0)

    time = np.arange(1, 500)
    seasonal_pattern = 50 * np.sin(2 * np.pi * time / 12)
    data = (
        5 + 0.5 * time + seasonal_pattern + np.random.normal(scale=60, size=len(time))
    )

    return pd.Series(
        data=data,
        index=pd.date_range(start="2020-01-01", periods=len(time), freq="D"),
    )


def get_time_series_with_outliers(n_outliers: int = 50) -> pd.Series:
    np.random.seed(0)

    df = get_noisy_time_series()
    outliers = np.random.choice(df.index, size=n_outliers, replace=False)
    df.loc[outliers] = df.loc[outliers] + np.random.normal(
        scale=300, size=len(outliers)
    )
    return df


def get_time_series_with_missing_values(
    n_missing: int = 50,
) -> Tuple[pd.Series, pd.Series]:
    df = get_noisy_time_series()
    missing_values = np.random.choice(df.index, n_missing, replace=False)
    df.loc[missing_values] = np.nan
    return df, missing_values
