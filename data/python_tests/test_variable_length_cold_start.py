import pytest
from thirdai import data

pytestmark = [pytest.mark.unit]


def default_augmentation(
    covering_min_length=3,
    covering_max_length=40,
    max_covering_samples=None,
    slice_min_length=3,
    slice_max_length=None,
    num_slices=2,
    add_whole_doc=True,
    prefilter_punctuation=True,
    strong_sample_num_words=3,
    word_removal_probability=0,
):
    return data.transformations.VariableLengthColdStart(
        strong_columns=["STRONG"],
        weak_columns=["WEAK"],
        label_column="LABELS",
        output_column="OUTPUT",
        covering_min_length=covering_min_length,
        covering_max_length=covering_max_length,
        max_covering_samples=max_covering_samples,
        slice_min_length=slice_min_length,
        slice_max_length=slice_max_length,
        num_slices=num_slices,
        add_whole_doc=add_whole_doc,
        prefilter_punctuation=prefilter_punctuation,
        strong_sample_num_words=strong_sample_num_words,
        word_removal_probability=word_removal_probability,
        seed=81,
    )


def test_vlcs_prefilter_punctuation():
    augmentation = default_augmentation()
    samples = augmentation.augment_single_row(
        "something is, strong text", "This is (weak) text."
    )

    forbidden_chars = [",", "(", ")", "."]
    for sample in samples:
        assert all(
            char not in sample for char in forbidden_chars
        ), "Forbidden character found in string"


@pytest.mark.parametrize("add_whole_doc", [True, False])
def test_vlcs_add_whole_doc(add_whole_doc):
    augmentation = default_augmentation(add_whole_doc=add_whole_doc)
    samples = augmentation.augment_single_row(
        "something is strong text", "This is weak text"
    )
    actually_added = "something is strong text This is weak text" in samples

    assert actually_added == add_whole_doc


def test_vlcs_covering_samples():
    augmentation = default_augmentation(
        covering_min_length=3, covering_max_length=5, num_slices=0, add_whole_doc=False
    )
    sentence = "This is weak text that has more than a certain amount of words"
    expected_words = sentence.split(" ")
    samples = augmentation.augment_single_row("", sentence)

    word_count = 0
    for sample in samples[:-1]:
        words = sample.strip().split(" ")
        assert 3 <= len(words) and len(words) <= 5
        for word in words:
            assert expected_words[word_count] == word
            word_count += 1
