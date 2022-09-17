from thirdai import bolt

model = bolt.SequentialClassifier(
    user=("user_col", 50), # (col name, n unique users)
    target=("target_col", 50), # (col name, n unique users)
    timestamp="timestamp_col",
    sequential=("sequential_col", 50, 5), # (col name, n unique classes, sequence length)
    dense_sequential=("dense_sequential_col", 5, 5, 5), # (col name, lookahead (days), lookback (days), period (days))
)