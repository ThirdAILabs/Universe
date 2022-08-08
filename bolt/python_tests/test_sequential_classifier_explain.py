from thirdai import bolt
import pytest
import os


def setup_module():
    with open("hello.txt", "w") as hello:
        hello.write(
            "discard_col,item_col,timestamp_col,target_col,text_col_1,text_col_2,text_col_3,cat_col_1,cat_col_2,cat_col_3,trackable_qty_1,trackable_qty_2,trackable_qty_3,trackable_cat_1,trackable_cat_2\n"
        )
        hello.write(
            "ignore,best class,2021-01-01,not_best,hello there,hello there too,hello there three,arbitrary_class_0,arbitrary_class_1,arbitrary_class_2,0.35,0.33,0.31,1,2\n"
        )


def define_sample_sequential_classifier():
    seq = bolt.SequentialClassifier(
        size="small",
        item=("item_col", 50),
        timestamp="timestamp_col",
        target=("target_col", 50),
        lookahead=0,  # num days to predict ahead
        lookback=10,  # num days to look back
        # Optional:
        period=1,  # expected num days between each record; period for clubbing data points together
        text=["text_col_1", "text_col_2", "text_col_3"],
        categorical=[
            ("cat_col_1", 50),
            ("cat_col_2", 50),
            ("cat_col_3", 50),
        ],
        trackable_qty=["trackable_qty_1", "trackable_qty_2", "trackable_qty_3"],
    )
    return seq


def test_number_of_times_column_names():
    """
    checking whether we are getting expected number of times of each column name.
    for trackable quantities we should get lookback times, for categorical 1 and for
    text we should get number of words in that text.
    """
    seq = define_sample_sequential_classifier()
    seq.train("hello.txt", epochs=1, learning_rate=0.001)
    temp = seq.explain("hello.txt")
    assert temp[0][0].count("item_col") == 1
    assert temp[0][0].count("timestamp_col") == 4
    assert temp[0][0].count("text_col_1") == 2
    assert temp[0][0].count("text_col_2") == 3
    assert temp[0][0].count("text_col_3") == 3
    assert temp[0][0].count("cat_col_1") == 1
    assert temp[0][0].count("cat_col_2") == 1
    assert temp[0][0].count("cat_col_3") == 1
    assert temp[0][0].count("trackable_qty_1") == 10
    assert temp[0][0].count("trackable_qty_2") == 10
    assert temp[0][0].count("trackable_qty_3") == 10


def test_total_percent_explanation_to_be_hundred():
    """
    The total sum of percent explanation must be so close to 100
    """
    seq = define_sample_sequential_classifier()
    seq.train("hello.txt", epochs=1, learning_rate=0.001)
    temp = seq.explain("hello.txt")
    total = 0
    for i in temp[1][0]:
        total += abs(i)

    assert total > 0.9999


def test_indices_within_block_within_range():
    """
    check whether we are getting indices within the block are within the correct range or not.
    """
    seq = define_sample_sequential_classifier()
    seq.train("hello.txt", epochs=1, learning_rate=0.001)
    temp = seq.explain("hello.txt")
    for i in range(len(temp[2][0])):
        if temp[0][0][i] == "item_col":
            assert temp[2][0][i] == 0
        if temp[0][0][i] == "timestamp_col":
            assert temp[2][0][i] >= 0 and temp[2][0][i] <= 73
        if (
            temp[0][0][i] == "text_col_1"
            or temp[0][0][i] == "text_col_2"
            or temp[0][0][i] == "text_col_3"
        ):
            assert temp[2][0][i] >= 0 and temp[2][0][i] <= 100000
        if (
            temp[0][0][i] == "cat_col_1"
            or temp[0][0][i] == "cat_col_2"
            or temp[0][0][i] == "cat_col_3"
        ):
            assert temp[2][0][i] == 0
        if (
            temp[0][0][i] == "trackable_qty_1"
            or temp[0][0][i] == "trackable_qty_2"
            or temp[0][0][i] == "trackable_qty_3"
        ):
            assert temp[2][0][i] >= 0 and temp[2][0][i] <= 9
