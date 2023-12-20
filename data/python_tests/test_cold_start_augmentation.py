import pytest
from thirdai import data

pytestmark = [pytest.mark.unit]


def get_real_dataset():
    """
    This function returns strong, weak and label lists for a few entries of a
    real text dataset that contains the kind of phrases / punctuation commonly
    encountered in practice.
    """
    strong_list = [
        "Wicker Basket",
        "Genkent Digital Food Thermometer",
        'Asherton 25.6" Patio Bar Stool (Set of 2)',
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
        "structural stretchers.",
    ]
    label_list = ["0", "1", "2"]
    return strong_list, weak_list, label_list


def create_test_column_map(text_columns, labels):
    """
    Creates a column map from input dictionaries of strings and lists of
    integers. The text_column dictionary entries become StringColumns with
    the name equal to their key, while the labels become a TokenArrayColumn
    named "labels." Note: text_columns should not contain "labels" as a key.
    Arguments:
        text_columns: Dictionary from string name to lists of strings.
        labels: Array-like of integer labels.
    """
    label_column = data.columns.StringColumn(labels)
    column_dict = {"labels": label_column}
    for name in text_columns.keys():
        column_dict[name] = data.columns.StringColumn(text_columns[name])

    columns = data.ColumnMap(column_dict)
    return columns


def apply_augmentation_and_unigrams(columns, augmentation):
    """
    Applies augmentation to columns, passed in as a ColumnMap, producing a new
    ColumnMap containing the self-supervised text inputs as a StringColumn
    named "data" and the integer labels "labels." The reason we apply
    unigrams is to get a more easily testable output, as currently the
    ColumnMap does not support introspection.
    """
    new_columns = augmentation(columns)

    featurizer = data.transformations.Text(
        input_column="data", output_indices="unigrams", dim=100000
    )
    columns = featurizer(new_columns)
    return columns


def test_duplicated_natural_separators():
    """
    Tests that the natural phrase generator behaves properly when presented
    with poorly-formatted inputs that contain multiple consecutive punctuation
    marks.
    """
    strong_list = [""]
    weak_list = [",,.,,.,.,,weak,,,...,, weak weak"]
    label_list = ["0"]
    augmentation = data.transformations.ColdStartText(
        strong_columns=["strong"],
        weak_columns=["weak"],
        output_column="data",
    )

    columns = create_test_column_map(
        {"strong": strong_list, "weak": weak_list}, label_list
    )
    new_columns = apply_augmentation_and_unigrams(columns, augmentation)

    # Verify that new_dataset consists of two entries, all having the same value
    data_list = new_columns["unigrams"].data()
    # Assert that we produced ["weak", "weak weak"] as the output data.
    # Because the data are shuffled internally, we can't count on the order
    # of terms in the output being the same as the input order.
    assert len(data_list) == 2
    if len(data_list[0]) < len(data_list[1]):
        one_word_list = data_list[0]
        two_word_list = data_list[1]
    else:
        one_word_list = data_list[1]
        two_word_list = data_list[0]
    assert len(one_word_list) == 1
    assert len(two_word_list) == 2
    assert one_word_list[0] == two_word_list[0] == two_word_list[1]


def test_long_input():
    """
    Tests that the natural phrase generator properly breaks phrases into
    correctly-sized chunks when the text contains no natural delimiters.
    """
    strong_list = ["strong"]
    weak_list = ["a b c d e f g h i j k l m n o"]
    label_list = ["0"]

    augmentation = data.transformations.ColdStartText(
        strong_columns=["strong"],
        weak_columns=["weak"],
        output_column="data",
        weak_max_len=3,
        weak_min_len=3,
    )

    columns = create_test_column_map(
        {"strong": strong_list, "weak": weak_list}, label_list
    )
    new_columns = apply_augmentation_and_unigrams(columns, augmentation)

    for row in new_columns["unigrams"].data():
        assert len(row) == 4  # 1 for the strong words, 3 for the phrase.


def test_sample_strong_words():
    """
    Tests that the phrase generator samples the correct number of words from
    the strong phrase, when the strong sampling flag is enabled.
    """
    strong_list = ["extremely ridiculously strong language"]
    weak_list = ["a b c d e f g h i j k l m n o"]
    label_list = ["0"]

    augmentation = data.transformations.ColdStartText(
        strong_columns=["strong"],
        weak_columns=["weak"],
        output_column="data",
        weak_max_len=3,
        weak_min_len=3,
        strong_sample_num_words=2,
    )

    columns = create_test_column_map(
        {"strong": strong_list, "weak": weak_list}, label_list
    )
    new_columns = apply_augmentation_and_unigrams(columns, augmentation)

    for row in new_columns["unigrams"].data():
        assert len(row) == 5  # 2 chosen from strong, 3 from weak.


def test_shuffle_correct():
    """
    Tests that the shuffling algorithm correctly shuffles the labels and the
    phrases together.
    """
    strong_list = ["A", "B", "C", "D"]
    weak_list = ["x x, x x", "x, x, x", "x x x", "x x x x"]
    label_list = ["2", "1", "3", "4"]

    augmentation = data.transformations.ColdStartText(
        strong_columns=["strong"],
        weak_columns=["weak"],
        output_column="data",
    )
    # This will take all natural phrases: two 2-word phrases for the first
    # example, three 1-word phrases for the second, a 3-word phrase for the third
    # and a 4-word phrase for the fourth. We can check that shuffling is done
    # correctly (e.g. does not mix up label-phrase pairs) by checking that each
    # phrase length is equal to its label, plus one for the strong phrase.

    columns = create_test_column_map(
        {"strong": strong_list, "weak": weak_list}, label_list
    )
    new_columns = apply_augmentation_and_unigrams(columns, augmentation)

    for label, unigrams in zip(
        new_columns["labels"].data(), new_columns["unigrams"].data()
    ):
        assert len(unigrams) == int(label) + 1


def test_sample_weak_words():
    """
    Tests that the phrase generator samples the correct number of words from
    the weak phrases, when the weak sampling flag is enabled.
    """
    strong_list = ["title"]
    weak_list = [
        "blah blah blah hashing blah blah lsh" " blah blah bloom filters blah blah"
    ]
    label_list = ["0"]

    num_examples_per_phrase = 17
    augmentation = data.transformations.ColdStartText(
        strong_columns=["strong"],
        weak_columns=["weak"],
        output_column="data",
        weak_sample_num_words=2,
        weak_sample_reps=num_examples_per_phrase,
    )
    # This will take all natural phrases, but downsample to just 2 tokens.

    columns = create_test_column_map(
        {"strong": strong_list, "weak": weak_list}, label_list
    )
    new_columns = apply_augmentation_and_unigrams(columns, augmentation)

    num_data = 0
    for row in new_columns["unigrams"].data():
        assert len(row) == 3  # 1 from strong, 2 chosen from weak.
        num_data += 1
    assert num_data == num_examples_per_phrase


def test_long_strong_phrase():
    """
    Tests that the strong phrase is cut to the correct number of words,
    when a maximum strong phrase length is provided.
    """
    strong_list = ["run on title that just goes on and on forever and ever"]
    weak_list = ["blah blah blah"]
    label_list = ["0"]

    augmentation = data.transformations.ColdStartText(
        strong_columns=["strong"],
        weak_columns=["weak"],
        output_column="data",
        strong_max_len=3,
    )

    columns = create_test_column_map(
        {"strong": strong_list, "weak": weak_list}, label_list
    )
    new_columns = apply_augmentation_and_unigrams(columns, augmentation)

    num_data = 0
    for row in new_columns["unigrams"].data():
        num_data += 1
        assert len(row) == 6  # 3 from strong, 3 from weak.
    assert num_data == 1


def test_multiple_weak_columns():
    """
    Tests that if we have multiple weak columns, they are concatenated to get
    the weak phrase.
    """
    strong_list = ["NeurIPS Reviews"]
    weak_list_0 = [
        "From Reviewer 1: This paper is good and contains important"
        ", novel results. Therefore, I have decided to reject it."
    ]
    weak_list_1 = [
        "From Reviewer 2: I detest this paper so much that I did "
        "not even read it. Reject (with high confidence)."
    ]
    weak_list_2 = [
        "From Reviewer 3: I have provided a list of 437 papers that"
        " this paper did not cite. All of them were written by me"
    ]
    label_list = ["0"]

    augmentation = data.transformations.ColdStartText(
        strong_columns=["strong"],
        weak_columns=["weak_0", "weak_1", "weak_2"],
        output_column="data",
    )

    columns = create_test_column_map(
        {
            "strong": strong_list,
            "weak_0": weak_list_0,
            "weak_1": weak_list_1,
            "weak_2": weak_list_2,
        },
        label_list,
    )
    new_columns = apply_augmentation_and_unigrams(columns, augmentation)

    assert len(new_columns["unigrams"].data()) == 12


def test_multiple_strong_columns():
    """
    Tests that if we have multiple strong columns, they are concatenated to get
    the strong phrase.
    """
    strong_list_0 = ["hashing"]
    strong_list_1 = ["sketching"]
    strong_list_2 = ["sampling"]
    weak_list = [
        "These techniques are common components of randomized "
        "algorithms that trade accuracy for efficiency."
    ]
    label_list = ["0"]

    augmentation = data.transformations.ColdStartText(
        strong_columns=["strong_0", "strong_1", "strong_2"],
        weak_columns=["weak"],
        output_column="data",
        weak_sample_num_words=1,
    )

    columns = create_test_column_map(
        {
            "strong_0": strong_list_0,
            "strong_1": strong_list_1,
            "strong_2": strong_list_2,
            "weak": weak_list,
        },
        label_list,
    )
    new_columns = apply_augmentation_and_unigrams(columns, augmentation)

    for row in new_columns["unigrams"].data():
        assert len(row) == 4  # 3 from strong, 1 sampled from weak.


def test_real_input():
    """
    This is a behavioral test to check that the augmentation produces
    a reasonable output on real-world input data. We want to produce a
    reasonable number of phrases - not too many but not too few - and to
    confirm that 95% of the phrases meet our length requirements (there
    may be some stragglers that get cut off at the end of a weak text block).
    """
    strong_list, weak_list, label_list = get_real_dataset()
    augmentation = data.transformations.ColdStartText(
        strong_columns=["strong"],
        weak_columns=["weak"],
        output_column="data",
        strong_sample_num_words=2,
        weak_min_len=5,
        weak_max_len=10,
        weak_chunk_len=5,
    )

    expected_min_length = 5  # minimally, 0 from strong and 5 from smallest weak phrase.
    expected_max_length = 12  # 2 from strong and 10 from largest strong phrase.

    columns = create_test_column_map(
        {"strong": strong_list, "weak": weak_list}, label_list
    )
    new_columns = apply_augmentation_and_unigrams(columns, augmentation)

    num_data = 0
    num_valid_data = 0
    for row in new_columns["unigrams"].data():
        num_data += 1
        if expected_min_length <= len(row) <= expected_max_length:
            num_valid_data += 1
    # This assertion checks that we get more than 2 but less than 100 phrases
    # per row of input. If we are getting more than 100 phrases, this is a
    # problem as it results in a very big pre-training task (100x larger).
    # Note that there are 3 rows of input, so we verify that there are between
    # 6 and 300 phrases to ensure that the average number of phrases / row is
    # acceptable.
    assert 6 <= num_data <= 300
    assert num_valid_data / num_data > 0.95
