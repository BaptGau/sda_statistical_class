import json

import numpy as np
import pandas as pd

from demos.utlis import get_time_series, setup_plot

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def evaluate_models(
    models: dict, train: pd.DataFrame, validation: pd.DataFrame
) -> dict:
    """
    Evaluate the models on the validation set.

    Args:
        models (dict): Dictionary of models to evaluate.
        train (pd.DataFrame): Training data.
        validation (pd.DataFrame): Validation data.

    Returns:
        dict: Dictionary of models with their predictions on the validation set.
    """
    results = {}
    for name, model in models.items():
        model_fit = model.fit()
        predictions = model_fit.forecast(len(validation))
        results[name] = f"{np.nanmean(np.abs(train - predictions))}:.2f"
    return results


if __name__ == "__main__":
    # todo : Quel est le meilleur modèle ? Faites une sélection de modèles en comparant les erreurs de prédiction
    # horizon de 36 points
    horizon = 36
    data = get_time_series()
    colors = setup_plot()

    train = data.iloc[:-horizon, :]
    validation = train.iloc[-horizon:, :]

    print(train, validation)

    test = data.iloc[-horizon:, :]

    models = {
        "SARIMAX": SARIMAX(train, order=(3, 1, 1), seasonal_order=(1, 1, 1, 12)),
        "HoltWinters": ExponentialSmoothing(
            train, trend="add", seasonal="add", seasonal_periods=12
        ),
    }

    model_perfs = evaluate_models(models, train, validation)
    print(json.dumps(model_perfs, indent=4, ensure_ascii=False))
