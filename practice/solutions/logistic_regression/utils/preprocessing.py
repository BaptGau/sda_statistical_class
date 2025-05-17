import logging

from sklearn.base import BaseEstimator, TransformerMixin

from artefacts.hypothesis_testing.data.input_parameters import (
    TtestInputTestParameters,
    AlternativeStudentHypothesis,
)
from artefacts.hypothesis_testing.statistical_tests.types.t_test import Ttest


class StatisticallySignificantFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.05):
        """
        Initialize the feature selector with a significance threshold.

        Parameters:
        - threshold (float): The p-value threshold below which features are kept.
        """
        self.threshold = threshold
        self.selected_features_ = None

    def fit(self, X, y):
        """
        Fit the feature selector by identifying statistically significant features.

        Parameters:
        - X (pd.DataFrame): The feature matrix.
        - y (pd.Series or np.array): The binary target vector.

        Returns:
        - self (StatisticallySignificantFeatureSelector): Fitted selector.
        """
        self.selected_features_ = []
        target_vector = y.values.ravel()

        for feature in X.columns:
            feature_vector = X.loc[:, feature].values
            group_0 = feature_vector[target_vector == 0]
            group_1 = feature_vector[target_vector == 1]

            t_test = Ttest()
            t_test.fit(
                group_0,
                group_1,
                threshold=self.threshold,
                input_parameters=TtestInputTestParameters(
                    equal_var=False, alternative=AlternativeStudentHypothesis.TWO_SIDED
                ),
            )

            if not t_test.is_null_hypothesis_true:
                logging.debug(f"Feature {feature} is statistically significant.")
                self.selected_features_.append(feature)

        return self

    def transform(self, X):
        """
        Transform the dataset by selecting only statistically significant features.

        Parameters:
        - X (pd.DataFrame): The feature matrix.

        Returns:
        - X_transformed (pd.DataFrame): Transformed feature matrix with selected features.
        """
        # Check if fit has been called
        if self.selected_features_ is None:
            raise ValueError("The transform method is called before fit.")

        return X[self.selected_features_]

    def fit_transform(self, X, y):
        """
        Fit to the data, then transform it.

        Parameters:
        - X (pd.DataFrame): The feature matrix.
        - y (pd.Series or np.array): The binary target vector.

        Returns:
        - X_transformed (pd.DataFrame): Transformed feature matrix with selected features.
        """
        return self.fit(X, y).transform(X)
