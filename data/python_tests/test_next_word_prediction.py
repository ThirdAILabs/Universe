import pytest
from thirdai import data


# TODO(Pratyush) Add serialization to NWP Dual Tokenizers
@pytest.mark.parametrize("serialize", [True, False])
@pytest.mark.unit
def test_next_word_prediction_dual_tokenizers(serialize):
    columns = data.ColumnMap({"text": data.columns.StringColumn(["Banana is yellow"])})

    transform = data.transformations.NextWordPredictionDualTokenizer(
        input_column="text",
        context_column="context",
        target_column="target",
        input_tokenizer="char-4",
        output_tokenizer="words",
    )
    if serialize:
        transform = data.transformations.deserialize(transform.serialize())

    columns = transform(columns)

    assert len(columns["context"].data()[0]) == 4
    assert len(columns["context"].data()[1]) == 7
    assert len(columns["target"].data()) == 2
