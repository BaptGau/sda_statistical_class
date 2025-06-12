from scipy.stats import levene

from mocks.hypothesis_testing.hypothesis_testing_mocks import mock_problems

def levene_test(sample_1: list[float], sample_2: list[float], alpha: float = 0.05) -> bool:
    """
    Perform Levene's test for equal variances.
    H0: The variances of the samples are equal.
    H1: The variances of the samples are not equal.

    Args:
        alpha (float): Significance level for the test (default is 0.05).
        sample_1 (list[float]): The first sample to compare variances.
        sample_2 (list[float]): The second sample to compare variances.

    Returns:
        (bool): True if the variance are equal, false otherwise.
    """
    stat, p_value = levene(sample_1, sample_2)
    if p_value < alpha:
        return False
    else:
        return True


if __name__ == "__main__":
    problem = mock_problems.get("variance_comparison")

    sample_1, sample_2 = problem.get_data()


    print("Are the variances equal?", levene_test(sample_1, sample_2))


    # stat, p_value = levene(sample_1, sample_2)
    #
    # print(f"Levene's test statistic: {stat} - p-value: {p_value}")
    #
