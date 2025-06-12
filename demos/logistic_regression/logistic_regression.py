import statsmodels.api as sm

from artefacts.logistic_regression.binary_classifier_evaluator import (
    evaluate_binary_classifier,
)
from demos.utlis import setup_plot, get_data, ProblemType, plot_classification

if __name__ == "__main__":

    colors = setup_plot()

    X, y = get_data(type=ProblemType.CLASSIFICATION)
    plot_classification(X=X, y=y, colors=colors)

    # fit linear regression
    modified_X = sm.add_constant(data=X)  # Add constant to estimate the intercept
    model = sm.Logit(y, modified_X)
    model = model.fit()

    probabilistic_preds = model.predict(modified_X)

    # print/plots results
    print(model.summary())
    plot_classification(X=X, y=y, colors=colors, clf=model)

    # evaluation
    evaluate_binary_classifier(
        y_true=y, y_score=probabilistic_preds, colors=colors
    )

    # avec sklearn
    # from sklearn.linear_model import LogisticRegression
    #
    # model = LogisticRegression(fit_intercept=True) # = sm.add_constant(data=X)
    #
    # model = model.fit(X, y)
    # y_score = model.predict_proba(X)[:, 1]
