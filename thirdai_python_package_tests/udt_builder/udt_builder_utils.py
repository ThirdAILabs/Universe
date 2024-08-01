import random
import string
import typing

from thirdai.bolt.udt_modifications import task_detector

NUM_ROWS = 100


def get_numerical_column(
    num_rows: int, min_val: float, max_val: float, create_floats: bool
):
    vals = []
    for _ in range(num_rows):
        if create_floats:
            # Generate a random float
            value = random.uniform(min_val, max_val)
        else:
            # Generate a random integer
            value = random.randint(int(min_val), int(max_val))
        vals.append(str(value))
    return vals


def get_int_categorical_column(
    num_rows: int, number_tokens_per_row, min_val, max_val, delimiter: str = " "
):
    categorical_column = []
    for _ in range(num_rows):
        # Generate a list of random integers for each row
        row_tokens = [
            str(random.randint(min_val, max_val)) for _ in range(number_tokens_per_row)
        ]
        # Join the list of integers with the specified delimiter to form the row string
        row_string = delimiter.join(row_tokens)
        categorical_column.append(row_string)
    return categorical_column


def get_string_categorical_column(
    num_rows: int,
    number_tokens_per_row,
    delimiter: str = " ",
    select_tokens_from: typing.List[str] = None,
):
    categorical_column = []
    for _ in range(num_rows):
        if select_tokens_from:
            # Select tokens from the provided set
            row = delimiter.join(
                random.choices(select_tokens_from, k=number_tokens_per_row)
            )
        else:
            # Generate random alphanumeric tokens if no set is provided
            row = delimiter.join(
                "".join(random.choices(string.ascii_letters + string.digits, k=5))
                for _ in range(number_tokens_per_row)
            )
        categorical_column.append(row)
    return categorical_column
