import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from artefacts.hypothesis_testing.data.input_parameters import (
    TtestInputTestParameters,
    AlternativeStudentHypothesis,
)
from artefacts.hypothesis_testing.statistical_tests.mann_whitney import MannWhitneyTest
from artefacts.hypothesis_testing.statistical_tests.t_test import Ttest


class TestStatArtifacts:
    def test_t_test_for_different_mean_samples(self):
        x = np.random.uniform(0, 1, 1000)
        y = np.random.uniform(10, 11, 1000)

        t_test = Ttest()
        input_parameters = TtestInputTestParameters(
            equal_var=True, alternative=AlternativeStudentHypothesis.TWO_SIDED
        )
        t_test.fit(x, y, input_parameters=input_parameters)

        assert (
            t_test.test_parameters.p_value <= 0.05
        ), "The p-value should be lesser than 0.05 for different means"
        assert (
            not t_test.is_null_hypothesis_true
        ), "The null hypothesis should be rejected for significantly different means"

    def test_t_test_for_same_means_samples(self):
        x = np.random.normal(0, 1, 1000)
        y = np.random.normal(0, 1, 1000)

        t_test = Ttest()
        t_test.fit(x, y, threshold=0.050)

        assert (
            t_test.test_parameters.p_value > 0.05
        ), "The p-value should be greater than 0.05 for similar means"
        assert (
            t_test.is_null_hypothesis_true
        ), "The null hypothesis should not be rejected when means are the same"

    def test_t_test_null_hypothesis_value(self):
        t_test = Ttest()
        assert (
            t_test.null_hypothesis == "Means of the samples are the same"
        ), "The null hypothesis text should specify means being equal"

    def test_t_test_raise_error_if_not_fitted(self):
        with pytest.raises(NotFittedError):
            t_test = Ttest()
            _ = (
                t_test.is_null_hypothesis_true
            )  # Accessing without fitting should raise error

    def test_mann_whitney_for_different_median(self):
        x = np.random.normal(0, 1, 1000)
        y = np.random.normal(100, 1, 1000)

        u_test = MannWhitneyTest()
        u_test.fit(x, y)

        assert (
            u_test.test_parameters.p_value < 0.05
        ), "The p-value should be lesser than 0.05 for different medians"
        assert (
            not u_test.is_null_hypothesis_true
        ), "The null hypothesis should be rejected for significantly different medians"

    def test_mann_whitney_for_same_median(self):
        x = np.random.normal(0, 1, 1000)
        y = np.random.normal(0, 5, 1000)

        u_test = MannWhitneyTest()
        u_test.fit(x, y, threshold=0.05)

        assert (
            u_test.test_parameters.p_value > 0.05
        ), "The p-value should be greater than 0.05 for similar medians"
        assert (
            u_test.is_null_hypothesis_true
        ), "The null hypothesis should not be rejected for similar medians"

    def test_y_test_null_hypothesis_value(self):
        u_test = MannWhitneyTest()
        assert (
            u_test.null_hypothesis == "Medians of the samples are the same"
        ), "The null hypothesis text should specify medians being equal"
