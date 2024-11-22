import statsmodels.api as sm
from artefacts.linear_regression.hypothesis_checker import HypothesisChecker
from demos.utlis import setup_plot, get_data, plot_data, ProblemType

if __name__ == "__main__":

    colors = setup_plot()

    X, y = get_data(type=ProblemType.REGRESSION)
    # plot_data(X=X, y=y, colors=colors)

    modified_X = sm.add_constant(data=X)  # Add constant to estimate the intercept
    model = sm.OLS(endog=y, exog=modified_X).fit()

    # print results summary
    print(model.summary())

    # retrieving regression line
    # preds = model.fittedvalues
    preds = model.predict(modified_X)  # both works

    # plot_data(X=X, y=y, colors=colors, preds=preds)

    hypothesis_check = HypothesisChecker.__from_given_input__(X=X, y=y, model=model)

    print(hypothesis_check.get_check_report())
