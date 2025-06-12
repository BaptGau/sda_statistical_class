from pandas import DataFrame


def extract_data_per_type(data: DataFrame) -> tuple[DataFrame, DataFrame]:
    """
    Extracts quantitative and qualitative data from the given DataFrame.

    Args:
        data (DataFrame): The input DataFrame containing both quantitative and qualitative data.

    Returns:
        tuple[DataFrame, DataFrame]: A tuple containing two DataFrames:
            - The first DataFrame contains only quantitative data.
            - The second DataFrame contains only qualitative data.
    """
    quantitative_data = data.select_dtypes(include=["number"])
    qualitative_data = data.select_dtypes(exclude=["number"])
    return quantitative_data, qualitative_data