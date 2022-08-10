from thirdai import bolt

train_file = "/share/benito/netflix/netflix_leave_10_out_train.csv"
test_file = "/share/benito/netflix/netflix_leave_10_out_test.csv"
predict_file = "/share/benito/netflix/netflix_leave_10_out_predictions.csv"

user_tup = ("userId", 301898)
movie_tup = ("movieId", 17770)


def seq_without_temporal_feats():
    return bolt.SequentialClassifier(
        size="xs",
        track_per=user_tup,
        timestamp="timestamp",
        target=movie_tup,
        lookahead=0,  # num days to predict ahead.
        lookback=14,  # num days to look back
        metrics=["hit_ratio_at_1", "hit_ratio_at_5", "hit_ratio_at_10", "hit_ratio_at_25", "hit_ratio_at_50", "hit_ratio_at_100"]
    )

def seq_with_temporal_feats():
    return bolt.SequentialClassifier(
        size="xs",
        track_per=user_tup,
        timestamp="timestamp",
        target=movie_tup,
        lookahead=0,  # num days to predict ahead.
        lookback=14,  # num days to look back
        trackable_cat=[movie_tup + (5,), movie_tup + (10,), movie_tup + (25,), movie_tup + (50,)],
        metrics=["hit_ratio_at_1", "hit_ratio_at_5", "hit_ratio_at_10", "hit_ratio_at_25", "hit_ratio_at_50", "hit_ratio_at_100"]
    )

def train_and_evaluate(seq):
    seq.train(train_file, epochs=3, learning_rate=0.0001,
              validation_filename=test_file)

def main():
    train_and_evaluate(seq_without_temporal_feats())
    train_and_evaluate(seq_with_temporal_feats())

if __name__ == "__main__":
    main()
