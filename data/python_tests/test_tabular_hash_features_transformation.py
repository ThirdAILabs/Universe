import pytest
from dataset_utils import get_random_str_column, verify_hash_distribution
from thirdai import data

pytestmark = [pytest.mark.unit]

NUM_ROWS = 10000
OUTPUT_RANGE = 100
NUM_WORDS = 5


def create_and_tabular_hash_random_dataset(use_pairgrams):
    num_cols = NUM_WORDS
    string_columns = [get_random_str_column(NUM_ROWS) for _ in range(num_cols)]

    columns = data.ColumnMap({f"column{i}": string_columns[i] for i in range(num_cols)})

    column_name_list = [f"column{i}" for i in range(num_cols)]

    featurizer = data.transformations.TransformationList(
        transformations=[
            data.transformations.StringHash(
                input_column=column_name,
                output_column=f"{column_name}_hashes",
            )
            for column_name in column_name_list
        ]
        + [
            data.transformations.TabularHashedFeatures(
                input_columns=[f"{col_name}_hashes" for col_name in column_name_list],
                output_column="tabular_hash_features",
                output_range=OUTPUT_RANGE,
                use_pairgrams=use_pairgrams,
            )
        ]
    )

    columns = featurizer(columns)

    return columns["tabular_hash_features"].data()


@pytest.mark.parametrize("pairgrams", [True, False])
def test_tabular_hash_features_transformation(pairgrams):
    pairgrams = create_and_tabular_hash_random_dataset(use_pairgrams=pairgrams)
    verify_hash_distribution(pairgrams, OUTPUT_RANGE)


