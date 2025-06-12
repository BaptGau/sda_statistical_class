import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from numpy._typing import ArrayLike
from sklearn.calibration import calibration_curve


def plot_confusion_matrix(
    y_true: ArrayLike, y_pred: ArrayLike
) -> None:
    """
    Trace une matrice de confusion en utilisant les valeurs réelles et prédites.

    Paramètres :
    -----------
    y_true : ArrayLike
        Les étiquettes réelles des classes.
    y_pred : ArrayLike
        Les étiquettes prédites par le modèle.

    Affichage :
    -----------
    Affiche une matrice de confusion avec annotations de comptage, de pourcentage,
    et avec une distinction entre les catégories de prédiction correctes (VP, VN)
    et incorrectes (FP, FN).
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    cf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred)
    group_names = ["Vrai négatif", "Faux positif", "Faux négatif", "Vrai positif"]
    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = [
        "{0:.2%}".format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)
    ]
    labels = [
        f"{v1}\n{v2}\n{v3}"
        for v1, v2, v3 in zip(group_names, group_counts, group_percentages)
    ]
    labels = np.asarray(labels).reshape(2, 2)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cf_matrix, annot=labels, fmt="", cmap="mako")
    plt.xlabel("Valeurs prédites")
    plt.ylabel("Vraies valeurs")
    plt.title("Matrice de confusion", fontweight="bold")
    plt.show()


def print_binary_classification_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> None:
    """
    Affiche les métriques de classification binaire basiques : précision,
    rappel, score F1 et exactitude.

    Paramètres :
    -----------
    y_true : ArrayLike
        Les étiquettes réelles des classes.
    y_pred : ArrayLike
        Les étiquettes prédites par le modèle.

    Affichage :
    -----------
    Affiche les scores d'exactitude, précision, rappel et F1 sur la console.
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    print(f"Accuracy: {accuracy_score(y_true, y_pred):.2f}")
    print(f"Precision: {precision_score(y_true, y_pred):.2f}")
    print(f"Recall: {recall_score(y_true, y_pred):.2f}")
    print(f"Score F1: {f1_score(y_true, y_pred):.2f}")


def plot_roc_curve(y_true: ArrayLike, y_score: ArrayLike, colors: list[str]) -> None:
    """
    Trace la courbe ROC et les distributions de probabilités prédites pour les
    classes positives et négatives.

    Paramètres :
    -----------
    y_true : ArrayLike
        Les étiquettes réelles des classes.
    y_score : ArrayLike
        Les probabilités prédites pour la classe positive.
    colors : list[str]
        Liste de couleurs pour personnaliser les courbes et les tracés.

    Affichage :
    -----------
    Affiche un graphique en deux parties : histogramme des prédictions et
    courbe ROC avec l’AUC.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    tn_preds = y_score[y_true == 0]
    tp_preds = y_score[y_true == 1]

    # Plot distributions
    axes[0].hist(tn_preds, color=colors[0], alpha=0.7, label="TN", density=True)
    axes[0].hist(tp_preds, color=colors[1], alpha=0.7, label="TP", density=True)
    axes[0].set_title("Résultat du modèle de régression logistique")
    axes[0].set_xlabel("Probabilité prédite")
    axes[0].axvline(x=0.5, color=colors[2], linestyle="--", label="Seuil (0.5)")
    axes[0].legend(loc="upper right")
    axes[0].grid(True)

    # Calculate ROC curve
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y_true=y_true, y_score=y_score)
    from sklearn.metrics import auc

    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    axes[1].plot(
        fpr, tpr, color=colors[3], lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})"
    )
    axes[1].plot([0, 1], [0, 1], color=colors[2], lw=2, linestyle="--")
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.0])
    axes[1].set_xlabel("False Positive Rate (FPR)")
    axes[1].set_ylabel("True Positive Rate (TPR)")
    axes[1].set_title("ROC Curve")
    axes[1].legend(loc="lower right")
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


def plot_calibration_curve(
    y_true: ArrayLike, y_score: ArrayLike, colors: list[str]
) -> None:
    """
    Trace la courbe de calibration pour visualiser la précision des probabilités
    prédites par rapport aux résultats réels.

    Paramètres :
    -----------
    y_true : ArrayLike
        Les étiquettes réelles des classes.
    y_score : ArrayLike
        Les probabilités prédites pour la classe positive.
    colors : list[str]
        Liste de couleurs pour personnaliser la courbe de calibration.

    Affichage :
    -----------
    Affiche la courbe de calibration avec la ligne de calibration parfaite en
    pointillés pour comparaison.
    """
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_score, n_bins=10
    )

    # Step 5: Plot the calibration curve
    plt.figure(figsize=(12, 6))
    plt.plot(
        mean_predicted_value,
        fraction_of_positives,
        "s-",
        label="Logit Calibration",
        color=colors[0],
    )
    plt.plot([0, 1], [0, 1], label="Perfect Calibration", ls="--", color=colors[2])
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Curve")
    plt.legend()
    plt.grid(True)
    plt.show()
