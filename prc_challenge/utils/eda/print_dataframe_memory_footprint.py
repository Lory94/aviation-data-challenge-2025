import pandas as pd

def print_dataframe_memory_footprint(df: pd.DataFrame) -> None:
    value = pd.io.formats.info.DataFrameInfo(data=df, memory_usage="deep").memory_usage_string.strip()
    print(value)
