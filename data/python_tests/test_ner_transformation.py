import re

import pytest
import thirdai
from thirdai import data

pytestmark = [pytest.mark.unit]

TAGS = [
    data.transformations.NERLearnedTag("O"),
    data.transformations.NERLearnedTag("NAME"),
    data.transformations.NERLearnedTag("EMAIL"),
]

target_tokenizer = thirdai.dataset.NaiveSplitTokenizer(" ")


def test_inequal_number_of_tokens_and_tags():
    TAG_TRACKER = data.transformations.NerTagTracker(tags=TAGS, ignored_tags=set())

    ner_transformation = data.transformations.NerTokenizerUnigram(
        "source",
        "featurized_sentence",
        "target",
        dyadic_num_intervals=2,
        target_word_tokenizers=[target_tokenizer],
        feature_enhancement_config=data.transformations.NerFeatureConfig(
            True, True, True, True, True, True, True
        ),
        tag_tracker=TAG_TRACKER,
    )

    columns = data.ColumnMap(
        {
            "source": data.columns.StringArrayColumn([["I", "am", "Groot"]]),
            "target": data.columns.StringArrayColumn([["O", "O", "EMAIL", "NAME"]]),
        }
    )

    with pytest.raises(
        IndexError,
        match=re.escape("Mismatch between the number of tokens and tags in row 0"),
    ):
        transformed_columns = ner_transformation(columns)


def test_label_not_in_tag_map():
    TAG_TRACKER = data.transformations.NerTagTracker(tags=TAGS, ignored_tags=set())

    ner_transformation = data.transformations.NerTokenizerUnigram(
        "source",
        "featurized_sentence",
        "target",
        dyadic_num_intervals=2,
        target_word_tokenizers=[target_tokenizer],
        feature_enhancement_config=data.transformations.NerFeatureConfig(
            True, True, True, True, True, True, True
        ),
        tag_tracker=TAG_TRACKER,
    )

    columns = data.ColumnMap(
        {
            "source": data.columns.StringArrayColumn([["I", "am", "Groot"]]),
            "target": data.columns.StringArrayColumn([["O", "O", "RANDOM"]]),
        }
    )

    with pytest.raises(
        ValueError,
        match=re.escape("tag 'RANDOM' not found in the list of tags."),
    ):
        transformed_columns = ner_transformation(columns)
