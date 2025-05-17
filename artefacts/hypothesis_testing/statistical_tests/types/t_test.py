from typing import Optional

from numpy._typing import ArrayLike
from sklearn.exceptions import NotFittedError

from artefacts.hypothesis_testing.data.input_parameters import TtestInputTestParameters
from artefacts.hypothesis_testing.data.result_test_parameters import TestParameters
from artefacts.hypothesis_testing.interface.interface_stats_test import (
    InterfaceStatsTest,
)


class Ttest(InterfaceStatsTest):
    null_hypothesis = "Means of the samples are the same"

    def __init__(self):
        self._is_null_hypothesis_true: bool | None = None

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        threshold: float = 0.5,
        input_parameters: Optional[
            TtestInputTestParameters
        ] = TtestInputTestParameters(),
    ) -> "Ttest":
        """
        Fit the t-test.

        Args:
            X (ArrayLike): The first sample.
            y (ArrayLike): The second sample.
            threshold (float): The threshold to reject the null hypothesis.

        Returns:
            Ttest: The fitted t-test.
        """

        from scipy.stats import ttest_ind

        statistic, p_value = ttest_ind(
            X,
            y,
            equal_var=input_parameters.equal_var,
            alternative=input_parameters.alternative.value,
        )

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
