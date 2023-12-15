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
    stopword_removal_probability=0,
    stopword_insertion_probability=0,
    word_perturbation_probability=0,
    chars_replace_with_space=0,
    chars_deleted=0,
    chars_duplicated=0,
    chars_replace_with_adjacents=0,
    seed=81,
):
    return data.transformations.VariableLengthColdStart(
        strong_columns=["STRONG"],
        weak_columns=["WEAK"],
        output_column="OUTPUT",
        config=data.transformations.VariableLengthConfig(
            covering_min_length=covering_min_length,
            covering_max_length=covering_max_length,
            max_covering_samples=max_covering_samples,
            slice_min_length=slice_min_length,
            slice_max_length=slice_max_length,
            num_slices=num_slices,
            add_whole_doc=add_whole_doc,
            prefilter_punctuation=prefilter_punctuation,
            strong_sample_num_words=strong_sample_num_words,
            stopword_removal_probability=stopword_removal_probability,
            stopword_insertion_probability=stopword_insertion_probability,
            word_removal_probability=word_removal_probability,
            word_perturbation_probability=word_perturbation_probability,
            chars_replace_with_space=chars_replace_with_space,
            chars_deleted=chars_deleted,
            chars_duplicated=chars_duplicated,
            chars_replace_with_adjacents=chars_replace_with_adjacents,
        ),
        seed=seed,
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

    word_index = 0
    for i, sample in enumerate(samples):
        words = sample.strip().split(" ")

        for word in words:
            assert expected_words[word_index] == word
            word_index += 1

        is_last_sample = i == len(samples) - 1
        if not is_last_sample:
            # we don't check the last sample because it may sometimes have
            # slightly more than the max len
            length = len(words)
            assert 3 <= length and length <= 5


def test_vlcs_random_slices():
    augmentation = default_augmentation(
        slice_min_length=3,
        slice_max_length=5,
        num_slices=5,
        max_covering_samples=0,
        add_whole_doc=False,
    )
    sentence = "This is weak text that has more than a certain amount of words"
    samples = augmentation.augment_single_row("", sentence)

    for sample in samples:
        assert sample.strip() in sentence

        length = len(sample.strip().split(" "))
        assert 3 <= length and length <= 5


def test_vlcs_word_removal():
    augmentation = default_augmentation(
        slice_min_length=10000,
        slice_max_length=10000,
        num_slices=1,
        max_covering_samples=0,
        add_whole_doc=False,
        word_removal_probability=1.0,
    )
    sentence = "word " * 10000
    samples = augmentation.augment_single_row("", sentence)

    assert len(samples) == 0

    augmentation = default_augmentation(
        slice_min_length=10000,
        slice_max_length=10000,
        num_slices=1,
        max_covering_samples=0,
        add_whole_doc=False,
        word_removal_probability=0.5,
    )
    sentence = "word " * 10000
    samples = augmentation.augment_single_row("", sentence)

    assert len(samples) == 1
    assert 4500 < len(samples[0].split(" ")) and len(samples[0].split(" ")) < 5500


def test_vlcs_empty_weak_text():
    augmentation = default_augmentation()
    samples = augmentation.augment_single_row("strong text", "")
    assert len(samples) == 1


def test_vlcs_empty_both_text():
    augmentation = default_augmentation()
    samples = augmentation.augment_single_row("", "")
    assert len(samples) == 0


def test_vlcs_slice_min_larger_than_text():
    augmentation = default_augmentation(
        slice_min_length=4,
        max_covering_samples=0,
        num_slices=2,
        add_whole_doc=False,
    )
    samples = augmentation.augment_single_row("", "only three words")
    assert len(samples) == 2
    assert samples[0] == samples[1]


def test_vlcs_covering_min_larger_than_text():
    augmentation = default_augmentation(
        covering_min_length=4,
        num_slices=0,
        max_covering_samples=2,
        add_whole_doc=False,
    )
    samples = augmentation.augment_single_row("", "only three words")
    assert len(samples) == 1


def test_vlcs_min_not_greater_than_max():
    with pytest.raises(
        ValueError,
        match=r"covering_min_length must be <= covering_max_length.",
    ):
        default_augmentation(
            covering_min_length=4,
            covering_max_length=3,
        )


def test_vlcs_min_not_greater_than_max():
    with pytest.raises(
        ValueError,
        match=r"slice_min_length must be <= slice_max_length.",
    ):
        default_augmentation(
            slice_min_length=4,
            slice_max_length=3,
        )


def default_start_columns():
    return data.ColumnMap(
        {
            "STRONG": data.columns.StringColumn(
                [
                    "This is a first strong text row haha",
                    "second strong texttttttt woah",
                ]
            ),
            "WEAK": data.columns.StringColumn(
                [
                    "A paragraph with some shenanigans",
                    "Lots and lots of words for greater chance variability",
                ]
            ),
            "LABELS": data.columns.StringColumn(["0", "1"]),
        }
    )


def many_perturbations_augmentation(seed=81):
    return default_augmentation(
        word_removal_probability=0,
        stopword_removal_probability=0,
        stopword_insertion_probability=0,
        word_perturbation_probability=0.5,
        chars_replace_with_space=0,
        chars_deleted=0,
        chars_duplicated=0,
        chars_replace_with_adjacents=0,
        seed=seed,
    )


def test_vlcs_consistent_augmentations_with_seeds():
    augmentation1 = many_perturbations_augmentation()
    augmentation2 = many_perturbations_augmentation()

    columns = default_start_columns()
    columns1 = augmentation1(columns)

    columns = default_start_columns()
    columns2 = augmentation2(columns)

    assert columns1["OUTPUT"].data() == columns2["OUTPUT"].data()


def test_vlcs_different_augmentations_with_different_seeds():
    augmentation1 = many_perturbations_augmentation(seed=800)
    augmentation2 = many_perturbations_augmentation(seed=801)

    columns = default_start_columns()
    columns1 = augmentation1(columns)

    columns = default_start_columns()
    columns2 = augmentation2(columns)

    assert columns1["OUTPUT"].data() != columns2["OUTPUT"].data()


def test_vlcs_two_rows_with_same_augmentations_are_different():
    augmentation = many_perturbations_augmentation()

    columns = data.ColumnMap(
        {
            "STRONG": data.columns.StringColumn(
                [
                    "Some strong text",
                    "Some strong text",
                ]
            ),
            "WEAK": data.columns.StringColumn(
                [
                    "A paragraph with some shenanigans",
                    "A paragraph with some shenanigans",
                ]
            ),
            "LABELS": data.columns.StringColumn(["0", "1"]),
        }
    )
    columns = augmentation(columns)

    outputs = columns["OUTPUT"].data()
    print(outputs)
    deduplicated_outputs = set(outputs)

    assert len(outputs) == len(deduplicated_outputs)
