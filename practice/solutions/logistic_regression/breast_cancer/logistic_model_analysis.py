import statsmodels.api as sm

from artefacts.logistic_regression.binary_classifier_evaluator import evaluate_binary_classifier
from demos.utlis import setup_plot
from practice.solutions.logistic_regression.breast_cancer.preprocessing import (
    preprocess_data,
)
from practice.solutions.logistic_regression.utils.utils import load_breast_cancer_data

TARGET_NAME = "has_cancer"

if __name__ == "__main__":
    X, y = load_breast_cancer_data()
    significant_features, y = preprocess_data(X, y)

    modified_X = sm.add_constant(significant_features)

    fitted_model = sm.Logit(y, modified_X).fit()
    print(fitted_model.summary())

    # results analysis
    colors = setup_plot()
    probabilistic_preds = fitted_model.predict(modified_X)
    target_vector = y.loc[:, TARGET_NAME].values

    # evaluation
    evaluate_binary_classifier(
        y_true=target_vector, y_score=probabilistic_preds, colors=colors
    )
