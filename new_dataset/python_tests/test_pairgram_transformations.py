import pytest
from thirdai import new_dataset as dataset

pytestmark = [pytest.mark.unit]

def get_sentence_str_column(col_length):
    return dataset.columns.StringColumn([f"value{i} value{i} value{i+1}" for i in range(col_length)])

# tests that if 
def test_sentence_unigram_deduplication():
col_length = 10000
column = get_sentence_str_column(col_length)

columns = dataset.ColumnMap({"sentence": column})

output_range = 100
featurizer = dataset.FeaturizationPipeline(
    transformations=[
        dataset.transformations.SentenceUnigram(
            input_column="sentence",
            output_column="deduplicated",
            output_range=output_range,
            deduplicate=True,
        ),
        dataset.transformations.SentenceUnigram(
            input_column="sentence",
            output_column="not_deduplicated",
            output_range=output_range,
            deduplicate=False,
        )
    ]
)

columns = featurizer.featurize(columns)
deduped_dataset = columns.convert_to_dataset(["deduplicated"], batch_size=col_length)
not_deduped_dataset = columns.convert_to_dataset(["not_deduplicated"], batch_size=col_length)

    for i in range(col_length):


    for i in range(10):
        print(dataset_object2[0][i])



