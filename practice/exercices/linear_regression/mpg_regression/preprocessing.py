from collections import OrderedDict
from typing import Tuple

import seaborn as sns
from pandas import DataFrame, Series

from practice.exercices.linear_regression.mpg_regression.tools.preprocessing_steps.nan_dropper import (
    NanDropperPreprocessor,
)
from practice.solutions.linear_regression.mpg_regression.statistical_approach.preprocessing.colinear_features_cleaner import (
    ColinearFeatureCleanerPreprocessor,
)
from practice.solutions.linear_regression.mpg_regression.statistical_approach.preprocessing.column_dropper_preprocessor import (
    ColumnDropperPreprocessor,
)
from practice.solutions.linear_regression.mpg_regression.statistical_approach.preprocessing.preprocessing_pipeline import (
    PreprocessingPipeline,
)
from practice.solutions.linear_regression.mpg_regression.statistical_approach.preprocessing.qualitative_column_preprocessor import (
    QualitativeColumnsOneHotEncodingPreprocessor,
)
from practice.solutions.linear_regression.mpg_regression.statistical_approach.preprocessing.target_splitter_preprocessor import (
    TargetSplitterPreprocessor,
)


def preprocess_data(data: DataFrame) -> Tuple[DataFrame, Series]:
    pipeline_steps = OrderedDict(
        [
            ("nan_dropper", NanDropperPreprocessor()),
            ("column_dropper", ColumnDropperPreprocessor()),
            ("qualitative_encoder", QualitativeColumnsOneHotEncodingPreprocessor()),
            ("colinear_feature_cleaner", ColinearFeatureCleanerPreprocessor()),
            ("target_splitter", TargetSplitterPreprocessor()),
        ]
    )

    arguments = {
        "nan_dropper": {},
        "column_dropper": {"column_name": "name"},
        "colinear_feature_cleaner": {
            "target_column": "mpg",
            "threshold": 0.99,
        },
        "qualitative_encoder": {"columns": ["origin"]},
        "target_splitter": {"target_column": "mpg"},
    }

    pipeline = PreprocessingPipeline(
        preprocessors=pipeline_steps, preprocessor_kwargs=arguments
    )

    return pipeline.fit_transform(X=data)


if __name__ == "__main__":
    data = sns.load_dataset(name="mpg")
    print(data.info())

    X, y = preprocess_data(data=data)
    print(X.info())
