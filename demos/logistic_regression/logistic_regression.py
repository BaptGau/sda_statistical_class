import statsmodels.api as sm

from artefacts.logistic_regression.binary_classifier_evaluator import (
    BinaryClassifierEvaluator,
)
from demos.utlis import setup_plot, get_data, ProblemType, plot_classification

if __name__ == "__main__":

    colors = setup_plot()

    X, y = get_data(type=ProblemType.CLASSIFICATION)
    plot_classification(X=X, y=y, colors=colors)

    # fit linear regression
    modified_X = sm.add_constant(data=X)  # Add constant to estimate the intercept
    model = sm.Logit(y, modified_X).fit()

    probabilistic_preds = model.predict(modified_X)

    # print/plots results
    print(model.summary())
    plot_classification(X=X, y=y, colors=colors, clf=model)

    # evaluation
    BinaryClassifierEvaluator().evaluate_classifier(
        y_true=y, y_score=probabilistic_preds, colors=colors
    )

    # avec sklearn
    # from sklearn.linear_model import LogisticRegression
    #
    # model = LogisticRegression(fit_intercept=True)
    #
    # model = model.fit(X, y)
    # y_score = model.predict_proba(X)[:, 1]
    #
    # BinaryClassifierEvaluator().evaluate_classifier(
    #     y_true=y, y_score=y_score, colors=colors
    # )
