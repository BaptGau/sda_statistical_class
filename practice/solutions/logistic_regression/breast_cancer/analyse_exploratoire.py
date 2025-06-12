import numpy as np
from pandas import concat

from demos.utlis import setup_plot
from practice.solutions.linear_regression.mpg_regression.analyse_exploratoire import (
    GlobalAnalysis,
    QuantitativeAnalysis,
)
from practice.solutions.logistic_regression.utils.utils import (
    load_breast_cancer_data,
    pairplots_on_features,
)

if __name__ == "__main__":
    colors = setup_plot()

    # Load the dataset
    X, y = load_breast_cancer_data()

    quantitative_data = X.select_dtypes(include=np.number)
    qualitative_data = X.select_dtypes(include="object")

    # global analysis of the data
    GlobalAnalysis.print_info(data=X)
    GlobalAnalysis.print_nan_statistics(data=X)
    # No missing values

    # analysis of the quantitative variables
    QuantitativeAnalysis.print_describe(data=quantitative_data)

    # pairplots
    # pairplots_on_features(features=quantitative_data, y=y, colors=colors)
    # we can see some interesting linear separability between the features
