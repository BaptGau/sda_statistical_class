import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from numpy._typing import ArrayLike

from artefacts.hypothesis_testing.data.input_parameters import (
    TtestInputTestParameters,
    AlternativeStudentHypothesis,
)
from artefacts.hypothesis_testing.statistical_tests.types.t_test import Ttest
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

    t_test = Ttest()

    input_params = TtestInputTestParameters(
        equal_var=True, alternative=AlternativeStudentHypothesis.TWO_SIDED
    )

    print(f"H0: {t_test.null_hypothesis}")
    t_test.fit(X=x, y=y)
    print(f"Is H0 true: {t_test.is_null_hypothesis_true}")

    print(f"Test parameters: {t_test.test_parameters.__dict__}")
