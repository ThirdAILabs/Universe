from thirdai import bolt
import pytest
import os


@pytest.mark.integration
def test_sequential_classifier_api():
    with open("hello.txt", "w") as hello:
        hello.write(
            "discard_col,item_col,timestamp_col,target_col,text_col_1,text_col_2,text_col_3,cat_col_1,cat_col_2,cat_col_3,trackable_qty_1,trackable_qty_2,trackable_qty_3,trackable_cat_1,trackable_cat_2\n"
        )
        hello.write(
            "ignore,best class,2021-01-01,not_best,hello there,hello there too,hello there three,arbitrary_class_0,arbitrary_class_1,arbitrary_class_2,0.35,0.33,0.31,1,2\n"
        )

    seq = bolt.SequentialClassifier(
        size="small",
        item=("item_col", 50),
        timestamp="timestamp_col",
        target=("target_col", 50),
        horizon=0,  # num days to predict ahead
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
        trackable_cat=[("trackable_cat_1", 50, 10)],
    )

    seq.train("hello.txt", epochs=5, learning_rate=0.001)
    seq.predict("hello.txt", "hello_predict.txt")

    os.remove("hello.txt")
    os.remove("hello_predict.txt")
