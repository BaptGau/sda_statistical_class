import numpy as np
import seaborn as sns

import statsmodels.api as sm
from matplotlib import pyplot as plt
from pandas import DataFrame
from statsmodels.regression.linear_model import RegressionResults

from artefacts.linear_regression.hypothesis_checker import check_all_hypotheses, format_check_report
from demos.utlis import setup_plot
from practice.solutions.linear_regression.mpg_regression.analyse_exploratoire import (
    QuantitativeAnalysis,
)
from practice.solutions.linear_regression.mpg_regression.data_preprocessing import (
    preprocess_data,
)


def plot_residuals_density(residuals: DataFrame, colors: list):
    plt.figure(figsize=(12, 8))
    sns.kdeplot(residuals, fill=True, color=colors[1], alpha=0.6)
    plt.axvline(
        np.mean(residuals),
        color=colors[0],
        linestyle="--",
        label="Moyenne des résidus",
    )
    p = plt.title("Distribution des résidus", fontweight="bold")
    plt.grid(True)
    sns.kdeplot(
        np.random.normal(np.mean(residuals), np.std(residuals), 10000),
        color=colors[2],
        linestyle="--",
        label="Distribution normale",
    )
    plt.legend()
    plt.show()


def plot_homoscedasticity(residuals: DataFrame, preds: DataFrame, colors: list[str]):
    import statsmodels.stats.api as sms
    from statsmodels.compat import lzip

    name = ["F statistic", "p-value"]
    test = sms.het_goldfeldquandt(residuals, X)
    stat, p = lzip(name, test)[1]

    plt.figure(figsize=(12, 8))
    plt.scatter(preds, residuals, color=colors[1], alpha=0.8)
    plt.xlabel("Valeurs prédites par le modèle")
    plt.ylabel("Résidus")
    plt.axhline(0, color=colors[0], linestyle="--")
    plt.title(
        f"Résidus vs. valeurs prédites - Gold-Felquant p-value: {p:.2f}",
        fontweight="bold",
    )
    plt.show()


def plot_residuals_autocorrelation(residuals: DataFrame):
    sm.graphics.tsa.plot_acf(residuals, lags=40, title="Autocorrélation des résidus")
    plt.grid(True)
    plt.show()


def plot_prediction_intervals(
    X, y, model: RegressionResults, colors: list[str], alpha=0.05
):
    # Calculate the standard error of the prediction
    predictions = model.get_prediction(X)
    prediction_summary = predictions.summary_frame(alpha=alpha)
    lower_bound = prediction_summary["obs_ci_lower"]
    upper_bound = prediction_summary["obs_ci_upper"]

    # Plot the data, fitted line, and prediction intervals
    plt.figure(figsize=(12, 8))
    plt.scatter(X.loc[:, "weight"], y, color=colors[0], label="Observed Data")

    # Determine colors for the error bars
    colors_error = [
        (
            colors[3]
            if lower_bound.iloc[i] <= y.iloc[i] <= upper_bound.iloc[i]
            else "salmon"
        )
        for i in range(len(y))
    ]

    # Plot error bars
    for i in range(len(y)):
        plt.errorbar(
            X.loc[:, "weight"].iloc[i],
            model.fittedvalues.iloc[i],
            yerr=[
                [model.fittedvalues.iloc[i] - lower_bound.iloc[i]],
                [upper_bound.iloc[i] - model.fittedvalues.iloc[i]],
            ],  # lower and upper bounds
            fmt="o",  # marker style
            alpha=0.7,
            color=colors[1],  # Color of the marker
            ecolor=colors_error[i],  # Color of the error bar based on the condition
            elinewidth=2,
            capsize=4,
        )

    # Add labels and legend
    plt.xlabel("weight")
    plt.ylabel("MPG")
    plt.legend()
    coverage = [color == colors[3] for color in colors_error].count(True) / len(y)
    plt.title(
        label=f"Prediction Intervals for OLS Regression Model - coverage: {coverage:.2f}",
        fontweight="bold",
    )
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    data = sns.load_dataset(name="mpg")
    X, y = preprocess_data(data=data)

    modified_X = sm.add_constant(data=X)  # Add constant to estimate the intercept
    model = sm.OLS(y, modified_X).fit()

    print(model.summary())

    # tests hypothesis through tests
    hypothesis_check = check_all_hypotheses(X=X, y=y, model=model)
    print(format_check_report(hypothesis_results=hypothesis_check))

    non_significant_features = model.pvalues[model.pvalues > 0.05].index

    modified_X = modified_X.drop(columns=non_significant_features)

    new_model = sm.OLS(y, modified_X).fit()
    print(new_model.summary())

    # graphical check of the hypothesis
    colors = setup_plot()

    plot_residuals_density(residuals=new_model.resid, colors=colors)
    plot_homoscedasticity(
        residuals=new_model.resid, preds=new_model.fittedvalues, colors=colors
    )
    plot_residuals_autocorrelation(residuals=new_model.resid)
    QuantitativeAnalysis.plot_linear_correlation(data=modified_X, colors=colors)

    # plot results
    print("Coef associated to each variable :")
    print(new_model.params)

    plot_prediction_intervals(X=modified_X, y=y, model=new_model, colors=colors)
