# Plot prediction results with MAPIE Jackknife estimation
import numpy as np
from mapie.metrics import regression_coverage_score
from mapie.regression import MapieRegressor
from matplotlib import pyplot as plt
from sklearn.model_selection import learning_curve


def plot_results_with_mapie(X_test, y_test, pipeline, colors: list[str]):
    # Wrap the final model with MAPIE for prediction intervals
    mapie = MapieRegressor(estimator=pipeline.named_steps["regressor"], method="plus")
    mapie.fit(pipeline.named_steps["preprocessor"].transform(X_test), y_test)

    # Generate predictions and prediction intervals
    y_pred, y_pis = mapie.predict(
        pipeline.named_steps["preprocessor"].transform(X_test), alpha=0.05
    )

    plt.figure(figsize=(12, 8))
    plt.scatter(
        X_test.loc[:, "weight"],
        y_test,
        alpha=0.7,
        color=colors[0],
        edgecolor="k",
        label="True values",
    )

    plt.scatter(
        X_test.loc[:, "weight"],
        y_pred,
        alpha=0.7,
        color=colors[1],
        edgecolor="k",
        label="Predictions",
    )

    # Plot the prediction intervals

    colors_error = [
        colors[3] if y_pis[i, 0, 0] <= y_test.iloc[i] <= y_pis[i, 1, 0] else "salmon"
        for i in range(len(y_test))
    ]

    # Plot predictions with error bars
    for i in range(len(y_pred)):
        plt.errorbar(
            X_test.loc[:, "weight"].iloc[i],
            y_pred[i],
            yerr=[
                [y_pred[i] - y_pis[i, 0, 0]],
                [y_pis[i, 1, 0] - y_pred[i]],
            ],  # lower and upper bounds
            fmt="o",  # marker style
            alpha=0.7,
            color=colors[1],  # Color of the marker
            ecolor=colors_error[i],  # Color of the error bar based on the condition
            elinewidth=2,
            capsize=4,
        )

    # Add labels and legend
    coverage = [color == colors[3] for color in colors_error].count(True) / len(y_test)
    plt.title(
        "Actual vs Predicted with Jackknife Prediction Intervals - Coverage: {:.2f}".format(
            coverage
        )
    )
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Calculate and print coverage score
    coverage = regression_coverage_score(y_test, y_pis[:, 0], y_pis[:, 1])
    print(f"Prediction Interval Coverage: {coverage:.2%}")


# Plot learning curve
def plot_learning_curve(pipeline, X, y, colors: list[str]):
    train_sizes, train_scores, test_scores = learning_curve(
        pipeline,
        X,
        y,
        cv=5,
        scoring="r2",
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5),
    )

    # Calculate mean and std
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(
        train_sizes,
        train_scores_mean,
        label="Training score",
        marker="o",
        color=colors[0],
    )
    plt.plot(
        train_sizes,
        test_scores_mean,
        label="Cross-validation score",
        marker="o",
        color=colors[1],
    )
    plt.title("Learning Curve")
    plt.xlabel("Training Size")
    plt.ylabel("R^2 Score")
    plt.legend()
    plt.grid(True)
    plt.show()
