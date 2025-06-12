import seaborn as sns

import statsmodels.api as sm

from demos.utlis import setup_plot
from practice.exercices.linear_regression.mpg_regression.preprocessing import (
    preprocess_data,
)
from practice.solutions.linear_regression.mpg_regression.ols_model_analysis import (
    plot_residuals_density,
    plot_homoscedasticity,
    plot_residuals_autocorrelation,
    plot_prediction_intervals,
)

if __name__ == "__main__":
    data = sns.load_dataset(name="mpg")
    print(data.info())

    X, y = preprocess_data(data=data)
    print(X.info())

    X = X.drop(
        columns=[
            "acceleration",
            "horsepower",
            "displacement",
            "cylinders",
            "model_year",
        ],
        axis=1,
    )

    colors = setup_plot()

    modified_X = X

    model = sm.OLS(endog=y, exog=modified_X).fit()

    print(model.summary())

    residuals = model.resid

    # plot_residuals_density(
    #     residuals=residuals,
    #     colors=colors,
    # )
    #
    # plot_homoscedasticity(
    #     residuals=model.resid, preds=model.fittedvalues, colors=colors
    # )
    #
    # plot_residuals_autocorrelation(residuals=residuals)
