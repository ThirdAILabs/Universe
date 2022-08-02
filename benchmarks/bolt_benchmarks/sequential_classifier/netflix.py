from thirdai import bolt
import os


def main():
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
        trackable_cat=[("trackable_cat_1", 50, 10)],
    )

    seq.train("hello.txt", epochs=5, learning_rate=0.001)
    seq.predict("hello.txt", "hello_predict.txt")

    os.remove("hello.txt")
    os.remove("hello_predict.txt")
