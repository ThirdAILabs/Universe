from thirdai import bolt

train_file = "/share/benito/movielens1m/train.csv"
test_file = "/share/benito/movielens1m/test.csv"
predict_file = "/share/benito/movielens1m/movielens1m_pred.csv"

user_tup = ("userId", 6040)
movie_tup = ("movieId",3706)


def seq_with_temporal_feats():
    return bolt.SequentialClassifier(
        size="xs",
        item=user_tup,
        timestamp="timestamp",
        target=movie_tup,
        lookahead=0,  # num days to predict ahead.
        lookback=14,  # num days to look back
        categorical=[user_tup],
        # Optional:
        # trackable_qty=[""],
        trackable_cat=[("movieId", 3706, 5), ("movieId", 3706, 10), ("movieId", 3706, 25), ("movieId", 3706, 50)],
        metrics=["categorical_accuracy", "precision_at_1", "precision_at_10", "precision_at_25", "precision_at_50", "precision_at_100"]
    )

def train_and_evaluate(seq):
    seq.train(train_file, epochs=10, learning_rate=0.0001,
              validation_filename=test_file)
    return seq.predict(test_file, predict_file)

def main():
    result_with_temporal_feats = train_and_evaluate(seq_with_temporal_feats())
    print("Accuracy with temporal feats:", result_with_temporal_feats)

if __name__ == "__main__":
    main()
