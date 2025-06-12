import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from numpy._typing import ArrayLike
from scipy.stats import ttest_ind

from demos.utlis import setup_plot


def plot_samples(x: ArrayLike, y: ArrayLike, colors: list[str]):
    # affichage des données
    plt.figure(figsize=(12, 8))
    sns.kdeplot(
        x, color=colors[0], fill=True, alpha=0.5, label="Densité du premier échantillon"
    )
    sns.kdeplot(
        y, color=colors[1], fill=True, alpha=0.5, label="Densité du second échantillon"
    )

    plt.xlabel("Valeurs")
    plt.ylabel("Densité")

    plt.title("Densité de deux échantillons")

    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    colors = setup_plot()
    # generation des données
    x = np.random.normal(0, 1, 100)
    y = np.random.normal(0, 5, 100)
    plot_samples(x=x, y=y, colors=colors)

    test_statistic, p_value = ttest_ind(x, y, equal_var=False)

    print(f"Test statistic: {test_statistic} | P-value: {p_value}")
