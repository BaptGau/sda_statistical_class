from numpy._typing import ArrayLike
from sklearn.exceptions import NotFittedError

from artefacts.hypothesis_testing.data.result_test_parameters import TestParameters
from artefacts.hypothesis_testing.interface.interface_stats_test import (
    InterfaceStatsTest,
)


class MannWhitneyTest(InterfaceStatsTest):
    null_hypothesis = "Medians of the samples are the same"

    def __init__(self):
        self._is_null_hypothesis_true: bool | None = None

    def fit(
        self, X: ArrayLike, y: ArrayLike, threshold: float = 0.5, **kwargs
    ) -> "MannWhitneyTest":
        """
        Fit the t-test.

        X (ArrayLike): The first sample.
        y (ArrayLike): The second sample.
        threshold (float): The threshold to reject the null hypothesis.
        """

        alternative = kwargs.get("alternative", "two-sided")

        from scipy.stats import mannwhitneyu

        statistic, p_value = mannwhitneyu(X, y, alternative=alternative)

        self.test_parameters = TestParameters.from_results(
            p_value=p_value, statistic=statistic
        )

        self._is_null_hypothesis_true = p_value >= threshold
        return self

    @property
    def is_null_hypothesis_true(self) -> bool:
        if self._is_null_hypothesis_true is not None:
            return self._is_null_hypothesis_true
        raise NotFittedError(
            "Test must be fitted before calling 'is_null_hypothesis_true'"
        )
