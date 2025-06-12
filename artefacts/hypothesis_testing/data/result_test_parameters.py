from dataclasses import dataclass


@dataclass
class TestParameters:
    p_value: float
    statistic: float
    conclusion: bool

    @staticmethod
    def from_results(p_value: float, statistic: float) -> "TestParameters":
        return TestParameters(p_value=p_value, statistic=statistic, conclusion=p_value < 0.05)
