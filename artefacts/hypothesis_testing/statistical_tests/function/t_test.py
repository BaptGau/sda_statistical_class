from scipy.stats import ttest_ind

def are_means_equal(sample1, sample2, alpha=0.05, equal_var=True):
    """
    Perform a two-sample t-test and return True if means are statistically equal.

    Parameters:
    - sample1, sample2: Lists or arrays of sample data
    - alpha: Significance level (default = 0.05)
    - equal_var: Assume equal variance in samples (default = True)

    Returns:
    - bool: True if we fail to reject the null hypothesis (means are equal),
            False if we reject it (means are different).
    """
    t_stat, p_value = ttest_ind(sample1, sample2, equal_var=equal_var)
    return p_value >= alpha


if __name__ == "__main__":
    s1 = [2.3, 2.9, 3.1, 4.0, 5.2]
    s2 = [2.1, 3.0, 3.0, 4.1, 5.0]

    if are_means_equal(s1, s2):
        print("Means are statistically equal.")
    else:
        print("Means are significantly different.")
