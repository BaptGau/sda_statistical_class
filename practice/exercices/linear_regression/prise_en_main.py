from demos.utlis import setup_plot, get_data, ProblemType

if __name__ == "__main__":

    colors = setup_plot()

    X, y = get_data(type=ProblemType.REGRESSION, n_features=5)
    # todo : Fit the linear regression on this, and anaylze the results
    print(X)