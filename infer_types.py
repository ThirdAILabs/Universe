import pandas as pd

POSSIBLE_DELIMETERS = [" ", ";", ":", "-", "\t", "|"]

def get_delimeter_ratios(column):
    delimeter_counts = {}
    for delimeter in POSSIBLE_DELIMETERS:
        delimeter_counts[delimeter] = sum(
            column.apply(lambda entry: entry.count(delimeter))
        )
    return delimeter_counts


def semantic_type_inference(filename, nrows=100):
    df = pd.read_csv(filename, nrows=nrows)
    semantic_types = []
    for column_name in df:
        column = df[column_name]

        if column.dtype == "float64":
            semantic_types.append((column_name, "numerical"))

        elif column.dtype == "int64":
            semantic_types.append((column_name, "categorical"))

        elif column.dtype == "object":

            # Try converting column to date to determine if it's a date column
            try:
                pd.to_datetime(column)
                semantic_types.append((column_name, "date"))
                continue
            except pd.errors.ParserError:
                pass

            delimeter_ratios = get_delimeter_ratios(column)

            # If the number of spaces per row is greater than a threshold then
            # it's text
            num_spaces = sum(column.apply(lambda entry: entry.count(" ")))
            if num_spaces / nrows > TEXT_SPACE_RATIO_HEURISTIC:
                semantic_types.append((column_name, "text"))
                continue

            str_col = df[column_name].astype("str")
