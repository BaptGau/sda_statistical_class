from pandas import DataFrame

def print_nan_statistics(data: DataFrame) -> None:
    print(" Missing values in the data ".center(50, "="))
    print(data.isna().sum())

def print_info(data: DataFrame) -> None:
    print(" Information about the data ".center(50, "="))
    print(data.info())