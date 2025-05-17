import numpy as np

from artefacts.hypothesis_testing.data.input_parameters import TtestInputTestParameters
from artefacts.hypothesis_testing.statistical_tests.types.t_test import Ttest
from mocks.hypothesis_testing.hypothesis_testing_mocks import mock_problems
from practice.solutions.hypothesis_testing.utils.plotter import plot_two_samples

if __name__ == "__main__":
    problem = mock_problems.get("binary_variable_comparison")
    performance, indicatrice = problem.get_data()
    print(problem.problem_statement)

    avec_formation = performance[np.where(indicatrice == "ont suivi une formation")]
    sans_formation = performance[
        np.where(indicatrice == "n'ont pas suivi de formation")
    ]

    plot_two_samples(
        sample_1=avec_formation,
        sample_2=sans_formation,
        sample_1_label="Avec formation",
        sample_2_label="Sans formation",
        figure_title="Comparaison des scores de performance",
        x_label="Scores de performance",
    )

    t_test = Ttest()
    t_test.fit(
        X=avec_formation,
        y=sans_formation,
        input_parameters=TtestInputTestParameters(equal_var=False),
    )
    print(f"Les moyennes sont identiques: {t_test.is_null_hypothesis_true}")
