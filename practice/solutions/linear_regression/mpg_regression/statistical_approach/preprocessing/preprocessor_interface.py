from abc import ABC, abstractmethod

from pandas import DataFrame


class PreprocessorInterface(ABC):
    @abstractmethod
    def fit(self, X: DataFrame, **kwargs) -> "PreprocessorInterface":
        pass

    @abstractmethod
    def transform(self, X: DataFrame) -> DataFrame:
        pass

    @abstractmethod
    def fit_transform(self, X: DataFrame, **kwargs) -> DataFrame:
        pass
