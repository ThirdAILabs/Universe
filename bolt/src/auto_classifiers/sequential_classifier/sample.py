from thirdai import bolt

model = bolt.SequentialClassifier(
    user=("user_col", 50), # (col name, n unique users)
    target=("target_col", 50), # (col name, n unique users)
    timestamp="timestamp_col",
    sequential=[("sequential_col", 50, 5)], # List of (col name, n unique classes, sequence length)
    dense_sequential=[("dense_sequential_col", 0, 1, 1)], # List of (col name, lookahead (days), lookback (days), period (days))
)