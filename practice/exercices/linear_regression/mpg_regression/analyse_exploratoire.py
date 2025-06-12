import seaborn as sns

from practice.exercices.linear_regression.mpg_regression.tools.analysis.qualitative_analysis import (
    print_modalities_number,
    plot_modalities_effect_on_target,
)
from practice.exercices.linear_regression.mpg_regression.tools.analysis.quantitatives_analysis import (
    print_describe,
    plot_linear_correlation,
    plot_pairplot,
)
from practice.exercices.linear_regression.mpg_regression.tools.utils import (
    extract_data_per_type,
)

if __name__ == "__main__":
    data = sns.load_dataset(name="mpg")

    # global analysis of the data
    # print_nan_statistics(data=data)
    # print_info(data=data)
    # only 6 missing values in the horsepower column
    # around 400 rows
    # 2 qualitive variables
    # 6 quantitative variables
    # Attention: model year

    quantitative_data, qualitative_data = extract_data_per_type(data=data)

    # print_describe(data=quantitative_data)

    # plot_linear_correlation(data=quantitative_data, colors=[f"black" for _ in range(quantitative_data.shape[1])])
    # Variables corrélées avec mpg: cool. Par contre multicolinéarité !!!
    # plot_pairplot(
    #     data=quantitative_data,
    #     hue=None,
    #     colors=[f"C{i}" for i in range(quantitative_data.shape[1])],
    # )

    # print_modalities_number(data=qualitative_data)

    # print(qualitative_data.name.value_counts())
    # too many names, can drop this column

    # plot_modalities_effect_on_target(
    #     data=data,
    #     target_column="mpg",
    #     qualitative_column="origin",
    #     colors=["C0", "C1", "C2"],
    # )

    # origin seems to have an effect on mpg (espcially USA vs JAPAN)
