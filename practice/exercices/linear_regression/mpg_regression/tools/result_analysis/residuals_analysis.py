import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pandas import Series


def plot_residuals_density(residuals: Series, colors: list):
    plt.figure(figsize=(12, 8))
    sns.kdeplot(residuals, fill=True, color=colors[1], alpha=0.6)
    plt.axvline(
        np.mean(residuals),
        color=colors[0],
        linestyle="--",
        label="Moyenne des résidus",
    )
    p = plt.title("Distribution des résidus", fontweight="bold")
    plt.grid(True)
    sns.kdeplot(
        np.random.normal(np.mean(residuals), np.std(residuals), 10000),
        color=colors[2],
        linestyle="--",
        label="Distribution normale",
    )
    plt.legend()
    plt.show()
