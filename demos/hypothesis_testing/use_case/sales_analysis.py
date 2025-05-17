from matplotlib import pyplot as plt
import seaborn as sns
from numpy._typing import ArrayLike
from scipy.stats import ttest_ind

from artefacts.hypothesis_testing.statistical_tests.types.t_test import Ttest
from demos.utlis import setup_plot
from mocks.hypothesis_testing.hypothesis_testing_mocks import mock_problems


def visualize_samples(
    pre_campagne_data: ArrayLike, post_campagne_data: ArrayLike, colors: list[str]
):
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))

    sns.kdeplot(
        pre_campagne_data,
        fill=True,
        color=colors[0],
        label="Ventes pré campagne",
        ax=axes[0],
        alpha=0.5,
    )
    sns.kdeplot(
        post_campagne_data,
        fill=True,
        color=colors[1],
        label="Ventes post campagne",
        ax=axes[0],
        alpha=0.5,
    )
    axes[0].set_xlabel("Montant des ventes")
    axes[0].legend()
    axes[0].grid(True)

    violins = axes[1].violinplot(
        [pre_campagne_data, post_campagne_data],
        vert=True,
        showmeans=False,
        showmedians=True,  # vertical box alignment
    )
    axes[1].set_xticks(
        [y + 1 for y in range(2)],
        labels=["Ventes pré campagne", rf"Ventes post campagne"],
        color="black",
        fontweight="bold",
    )

    for i, pc in enumerate(violins["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor("white")
        violins["cbars"].set_color(colors)
        violins["cmedians"].set_color(colors)
        violins["cmaxes"].set_color(colors)
        violins["cmins"].set_color(colors)
        axes[1].grid(True)

    plt.suptitle("Comparaison des ventes pré et post campagne", fontweight="bold")
    plt.show()


if __name__ == "__main__":
    colors = setup_plot()
    # generation des données

    problem = mock_problems.get("sales_comparison")
    pre_campagne_data, post_campagne_data = problem.get_data()

    print(problem.problem_statement)

    visualize_samples(
        pre_campagne_data=pre_campagne_data,
        post_campagne_data=post_campagne_data,
        colors=colors,
    )

    test_statistic, p_value = ttest_ind(
        pre_campagne_data, post_campagne_data, equal_var=True, alternative="two-sided"
    )

    print(f"Test statistic: {test_statistic} | P-value: {p_value}")

    t_test = Ttest()
    print(f"H0: {t_test.null_hypothesis}")

    t_test.fit(
        X=pre_campagne_data,
        y=post_campagne_data,
    )
    print(f"Is H0 true: {t_test.is_null_hypothesis_true}")

    print(f"Test parameters: {t_test.test_parameters.__dict__}")
