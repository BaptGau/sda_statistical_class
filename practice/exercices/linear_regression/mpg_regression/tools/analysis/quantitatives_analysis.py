from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import patches
from pandas import DataFrame


def print_describe(data: DataFrame) -> None:
    pd.set_option('display.max_columns', None)
    print(" Descriptive statistics of the data ".center(50, "="))
    print(data.describe())


def plot_linear_correlation(data: DataFrame, colors: list[str]) -> None:
    sns.set_style(style="whitegrid")
    corr = data.corr(method="pearson")
    mask = np.triu(np.ones_like(corr, dtype=bool))

    plt.figure(figsize=(12, 8))
    plt.title(
        label="Coefficient de corrélation linéaire entre les variables",
        fontsize=13,
        fontweight="bold",
    )
    sns.heatmap(corr, annot=True, cmap="mako", mask=mask, square=True, alpha=0.6)

    # adding rectangle
    ax = plt.gca()

    rect = patches.Rectangle(
        (0, 0), 1, data.shape[0], linewidth=1, edgecolor=colors[1], facecolor="none"
    )
    ax.add_patch(rect)

    plt.show()


def plot_pairplot(
    data: DataFrame, colors: list[str], hue: Optional[str] = None
) -> None:
    if hue:
        n_modalities = data.loc[:, hue].nunique()
        sns.pairplot(
            data,
            plot_kws={"alpha": 0.6},
            diag_kws={"fill": True, "alpha": 0.6},
            diag_kind="kde",
            kind="scatter",
            palette=colors[:n_modalities],
            hue=hue,
        )
    else:
        sns.pairplot(
            data,
            plot_kws={"alpha": 0.6, "color": colors[1]},
            diag_kws={"fill": True, "alpha": 0.6, "color": colors[0]},
            diag_kind="kde",
            kind="scatter",
        )
    plt.show()
