from matplotlib import pyplot as plt
import seaborn as sns
from pandas import DataFrame


def print_modalities_number(data: DataFrame) -> None:
    print(data.nunique(axis=0))


def plot_modalities_effect_on_target(
        data: DataFrame, target_column: str, qualitative_column: str, colors: list[str]
) -> None:

    palette = sns.color_palette(
        palette=colors, n_colors=data.loc[:, qualitative_column].nunique()
    )

    plt.figure(figsize=(12, 8))

    sns.kdeplot(
        data=data,
        x=target_column,
        hue=qualitative_column,
        palette=palette,
        fill=True,
        alpha=0.6,
    )
    for modality in data.loc[:, qualitative_column].unique():
        plt.axvline(
            x=data.loc[
                data.loc[:, qualitative_column] == modality, target_column
            ].mean(),
            color=palette[
                data.loc[:, qualitative_column].unique().tolist().index(modality)
            ],
            linestyle="--",
            label=f"{modality} mean",
        )

    plt.title(
        label=f"Effet des modalités de la variable {qualitative_column} sur la variable cible ({target_column})",
        fontsize=13,
        fontweight="bold",
    )
    plt.ylabel("Densité")
    plt.grid(True)
    plt.show()

