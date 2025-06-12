import logging

from pandas import DataFrame

from practice.solutions.linear_regression.mpg_regression.statistical_approach.preprocessing.preprocessor_interface import (
    PreprocessorInterface,
)


class NanDropperPreprocessor(PreprocessorInterface):
    def fit(self, X: DataFrame) -> "NanDropperPreprocessor":
        return self

    def transform(self, X: DataFrame) -> DataFrame:
        logger = logging.getLogger("preprocessing_logger")
        logger.debug(f"Dropping {X.isna().sum().max()} rows with NaN values")
        return X.dropna(axis=0)

    def fit_transform(self, X: DataFrame) -> DataFrame:
        return self.fit(X=X).transform(X=X)
