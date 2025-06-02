from abc import abstractmethod, ABC
from typing import Optional

from numpy._typing import ArrayLike


class InterfaceStatsTest(ABC):
    """
    Interface for statistical tests.
    Follows the skicit-learn API.
    """

    null_hypothesis: str

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def fit(
        self,
        X: ArrayLike,
        y: Optional[ArrayLike] = None,
        threshold: float = 0.5,
        *args,
        **kwargs
    ) -> "InterfaceStatsTest":
        pass

    @property
    @abstractmethod
    def is_null_hypothesis_true(self) -> bool:
        pass
