from mocks.hypothesis_testing.hypothesis_testing_mocks import mock_problems

if __name__ == "__main__":
    problem = mock_problems.get("sales_vs_ads")
    sample = problem.get_data()
    print(problem.problem_statement)
