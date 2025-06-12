from numpy._typing import ArrayLike

from artefacts.logistic_regression.evaluation_functions import (
    print_binary_classification_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_calibration_curve,
)


def evaluate_binary_classifier(
    y_true: ArrayLike,
    y_score: ArrayLike,
    colors: list[str],
    print_metrics: bool = True,
    show_confusion_matrix: bool = True,
    show_roc_curve: bool = True,
    show_calibration_curve: bool = True,
) -> None:
    """
    Évalue un modèle de classification binaire en affichant les métriques et graphiques sélectionnés.

    Args:
    ----------
    y_true : ArrayLike
        Les étiquettes réelles des classes.
    y_score: ArrayLike
        Les scores de probabilité prédits par le modèle.
    colors: list[str]
        Couleurs utilisées pour les graphiques.
    print_metrics : bool
        Affiche les métriques de classification (par défaut True).
    show_confusion_matrix : bool
        Affiche la matrice de confusion (par défaut True).
    show_roc_curve : bool
        Affiche la courbe ROC (par défaut True).
    show_calibration_curve : bool
        Affiche la courbe de calibration (par défaut True).
    """
    y_pred = y_score > 0.5

    if print_metrics:
        print_binary_classification_metrics(y_true=y_true, y_pred=y_pred)
    if show_confusion_matrix:
        plot_confusion_matrix(y_true=y_true, y_pred=y_pred)
    if show_roc_curve:
        plot_roc_curve(y_true=y_true, y_score=y_score, colors=colors)
    if show_calibration_curve:
        plot_calibration_curve(y_true=y_true, y_score=y_score, colors=colors)
