from mocks.hypothesis_testing.hypothesis_testing_mocks import mock_problems

if __name__ == "__main__":
    problem = mock_problems.get("service_ratings_comparison")
    sample_1, sample_2, sample_3 = problem.get_data()
    print(problem.problem_statement)