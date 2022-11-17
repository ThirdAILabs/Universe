import random
import string

import numpy as np
import pytest
from thirdai import data

pytestmark = [pytest.mark.unit]


def get_real_dataset():
    strong_list = [
        "Wicker Basket",
        "Genkent Digital Food Thermometer",
        "Asherton 25.6\" Patio Bar Stool (Set of 2)"
    ]
    weak_list = [
        "Get clutter in order with this storage basket with liner."
        "  From the kitchen to the bathroom to your home office, "
        "this storage bin works in any room in your home and its "
        "easy-grip handles help you get it there. You can feel good "
        "carrying it with its recycled and re-purposed parchment, and the "
        "tan liner adds an earthy tone to your aesthetic. Choose from three"
        " different sizes to find the basket that best meets your needs. ",
        "The unique fast reading system achieves an accurate temperature "
        "reading within 4 - 7 seconds. Ultra-clear screen LCD display. "
        "The foldaway probe is designed for safety and cleanliness. "
        "Intended Use: Deep fry; Meat; Oven  Instant Read: Yes  "
        "Automatic Shutoff: Yes",
        "Inspired by mid-century design, this patio bar stool showcases a "
        "curvaceous, one-piece seat with a full back and integrated arms, "
        "while a little flexibility offers added comfort. Mesh strap-style "
        "construction lets rainfall right through for quick-drying, while "
        "four gases injection-molded legs include a built-in footrest and "
        "structural stretchers."
    ]
    label_list = [0, 1, 2]
    return strong_list, weak_list, label_list


def apply_augmentation_and_unigrams(strong_list, weak_list, label_list,
        augmentation):
    # Applies augmentation to dataset, producing self-supervised "data" and 
    # "labels"
    label_list = np.array(label_list, dtype=np.uint32).reshape([-1, 1])
    strong_column = data.columns.StringColumn(strong_list)
    weak_column = data.columns.StringColumn(weak_list)
    label_column = data.columns.NumpySparseArrayColumn(
        array=label_list, dim=np.max(label_list)+1)

    columns = data.ColumnMap({"strong": strong_column,
                              "weak"  : weak_column,
                              "labels": label_column})

    new_columns = augmentation.apply(columns)

    featurizer = data.FeaturizationPipeline(
        transformations=[
            data.transformations.SentenceUnigram(
                input_column="data",
                output_column="unigrams",
                deduplicate=False,
                output_range=100000
            )
        ]
    )
    columns = featurizer.featurize(new_columns)
    return columns.convert_to_dataset(["unigrams"], batch_size=10)


def test_poorly_formatted_input():
    strong_list = [""]
    weak_list = [",,.,,.,.,,weak,,,...,, weak weak"]
    label_list = [0]

    augmentation = data.augmentations.ColdStartText(
        strong_columns=["strong"], weak_columns=["weak"],
        label_column="labels", output_column="data")

    new_dataset = apply_augmentation_and_unigrams(
        strong_list, weak_list, label_list, augmentation)
    # Verify that new_dataset consists of two entries, all having the same value
    
    data_list = []
    for batch in new_dataset:
        for row_id in range(len(batch)):
            row = batch[row_id]
            data_list.append(row.to_numpy()[0])
    # Assert that we produced ["weak", "weak weak"] as the output data.
    assert len(data_list) == 2
    assert len(data_list[0]) == 1
    assert len(data_list[1]) == 2
    assert data_list[0][0] == data_list[1][0] == data_list[1][1]


def test_long_input():
    strong_list = ["strong"]
    weak_list = ["a b c d e f g h i j k l m n o"]
    label_list = [0]

    augmentation = data.augmentations.ColdStartText(
        strong_columns=["strong"], weak_columns=["weak"],
        label_column="labels", output_column="data",
        weak_max_len=3, weak_min_len=3)

    new_dataset = apply_augmentation_and_unigrams(
        strong_list, weak_list, label_list, augmentation)
    # Verify that new_dataset consists of two entries, all having the same value
    
    data_list = []
    for batch in new_dataset:
        for row_id in range(len(batch)):
            row = batch[row_id].to_numpy()[0]
            assert len(row) == 4  # 1 for the strong words, 3 for the phrase.

def test_sample_strong_words():
    strong_list = ["extremely ridiculously strong language"]
    weak_list = ["a b c d e f g h i j k l m n o"]
    label_list = [0]

    augmentation = data.augmentations.ColdStartText(
        strong_columns=["strong"], weak_columns=["weak"],
        label_column="labels", output_column="data",
        weak_max_len=3, weak_min_len=3, strong_downsample_num=2)

    new_dataset = apply_augmentation_and_unigrams(
        strong_list, weak_list, label_list, augmentation)

    data_list = []
    for batch in new_dataset:
        for row_id in range(len(batch)):
            row = batch[row_id].to_numpy()[0]
            assert len(row) == 5  # 2 chosen from strong, 3 from weak.


def test_sample_weak_words():
    strong_list = ["title"]
    weak_list = ["blah blah blah hashing blah blah lsh"
                 " blah blah bloom filters blah blah"]
    label_list = [0]

    augmentation = data.augmentations.ColdStartText(
        strong_columns=["strong"], weak_columns=["weak"],
        label_column="labels", output_column="data",
        weak_downsample_num=2, weak_downsample_reps = 17)
    # This will take all natural phrases, but downsample to just 2 tokens.

    new_dataset = apply_augmentation_and_unigrams(
        strong_list, weak_list, label_list, augmentation)

    num_data = 0
    for batch in new_dataset:
        for row_id in range(len(batch)):
            row = batch[row_id].to_numpy()[0]
            assert len(row) == 3  # 1 from strong, 2 chosen from weak.
            num_data += 1
    assert num_data == 17


def test_long_strong_phrase():
    strong_list = ["run on title that just goes on and on forever and ever"]
    weak_list = ["blah blah blah"]
    label_list = [0]

    augmentation = data.augmentations.ColdStartText(
        strong_columns=["strong"], weak_columns=["weak"],
        label_column="labels", output_column="data",
        strong_max_len=3)

    new_dataset = apply_augmentation_and_unigrams(
        strong_list, weak_list, label_list, augmentation)

    num_data = 0
    for batch in new_dataset:
        for row_id in range(len(batch)):
            row = batch[row_id].to_numpy()[0]
            num_data += 1
            assert len(row) == 6  # 3 from strong, 3 from weak.
    assert num_data == 1


def test_real_input():
    strong_list, weak_list, label_list = get_real_dataset()
    augmentation = data.augmentations.ColdStartText(
        strong_columns=["strong"], weak_columns=["weak"],
        label_column="labels", output_column="data",
        strong_downsample_num=2, weak_min_len=5, weak_max_len=10,
        weak_chunk_len=5)
    # This is a behavioral test to check that the augmentation produces
    # something reasonable on real-world input data. We want to produce a
    # reasonable number of phrases - not too many but not too few - and to
    # confirm that 95% of the phrases meet our length requirements (there
    # may be some stragglers that get cut off at the end of a weak text block)

    min_length = 5  # minimally, 0 from strong and 5 from smallest weak phrase.
    max_length = 12  # 2 from strong and 10 from largest strong phrase.

    new_dataset = apply_augmentation_and_unigrams(
        strong_list, weak_list, label_list, augmentation)

    num_data = 0
    num_valid_data = 0
    for batch in new_dataset:
        for row_id in range(len(batch)):
            row = batch[row_id].to_numpy()[0]
            num_data += 1
            if min_length <= len(row) <= max_length:
                num_valid_data += 1
    assert 30 <= num_data <= 300
    assert num_valid_data / num_data > 0.95


def test_multiple_strong_columns():
    strong_list_0 = ["hashing"]
    strong_list_1 = ["sketching"]
    strong_list_2 = ["sampling"]
    weak_list = ["These techniques are common components of randomized "
                 "algorithms that trade accuracy for efficiency."]
    label_list = np.array([0], dtype=np.uint32).reshape([-1, 1])

    strong_column_0 = data.columns.StringColumn(strong_list_0)
    strong_column_1 = data.columns.StringColumn(strong_list_1)
    strong_column_2 = data.columns.StringColumn(strong_list_2)

    augmentation = data.augmentations.ColdStartText(
        strong_columns=["strong_0", "strong_1", "strong_2"],
        weak_columns=["weak"], label_column="labels", output_column="data",
        weak_downsample_num=1)

    weak_column = data.columns.StringColumn(weak_list)
    label_column = data.columns.NumpySparseArrayColumn(
        array=label_list, dim=np.max(label_list)+1)

    columns = data.ColumnMap({"strong_0": strong_column_0,
                              "strong_1": strong_column_1,
                              "strong_2": strong_column_2,
                              "weak"  : weak_column,
                              "labels": label_column})

    new_columns = augmentation.apply(columns)

    featurizer = data.FeaturizationPipeline(
        transformations=[
            data.transformations.SentenceUnigram(
                input_column="data",
                output_column="unigrams",
                deduplicate=False,
                output_range=100000
            )
        ]
    )
    columns = featurizer.featurize(new_columns)
    new_dataset = columns.convert_to_dataset(["unigrams"], batch_size=10)

    for batch in new_dataset:
        for row_id in range(len(batch)):
            row = batch[row_id].to_numpy()[0]
            assert len(row) == 4  # 3 from strong, 1 sampled from weak.

def test_multiple_weak_columns():
    strong_list = ["NeurIPS Reviews"]
    weak_list_0 = ["From Reviewer 1: This paper is good and contains important"
                   ", novel results. Therefore, I have decided to reject it."]
    weak_list_1 = ["From Reviewer 2: I detest this paper so much that I did "
                   "not even read it. Reject (with high confidence)."]
    weak_list_2 = ["From Reviewer 3: I have provided a list of 437 papers that"
                   " this paper did not cite. All of them were written by me"]
    label_list = np.array([0], dtype=np.uint32).reshape([-1, 1])

    augmentation = data.augmentations.ColdStartText(
        strong_columns=["strong"], weak_columns=["weak_0", "weak_1", "weak_2"],
        label_column="labels", output_column="data")

    strong_column = data.columns.StringColumn(strong_list)
    weak_column_0 = data.columns.StringColumn(weak_list_0)
    weak_column_1 = data.columns.StringColumn(weak_list_1)
    weak_column_2 = data.columns.StringColumn(weak_list_2)
    label_column = data.columns.NumpySparseArrayColumn(
        array=label_list, dim=np.max(label_list)+1)

    columns = data.ColumnMap({"strong": strong_column,
                              "weak_0": weak_column_0,
                              "weak_1": weak_column_1,
                              "weak_2": weak_column_2,
                              "labels": label_column})

    new_columns = augmentation.apply(columns)

    featurizer = data.FeaturizationPipeline(
        transformations=[
            data.transformations.SentenceUnigram(
                input_column="data",
                output_column="unigrams",
                deduplicate=False,
                output_range=100000
            )
        ]
    )
    columns = featurizer.featurize(new_columns)
    new_dataset = columns.convert_to_dataset(["unigrams"], batch_size=10)

    num_data = 0
    for batch in new_dataset:
        num_data += len(batch)
    # There are 12 total natural phrases in the three weak columns.
    assert num_data == 12

