from thirdai import bolt

train_file = "/Users/benitogeordie/Desktop/thirdai_datasets/movielens1m/train.csv"
test_file = "/Users/benitogeordie/Desktop/thirdai_datasets/movielens1m/test.csv"
predict_file = "/Users/benitogeordie/Desktop/thirdai_datasets/movielens1m/predictions.csv"

user_tup = ("userId", 6040)
movie_tup = ("movieId",3706)


def seq_with_temporal_feats():
    return bolt.SequentialClassifier(
        model_size="small",
        user=user_tup,
        timestamp="timestamp",
        target=movie_tup,
        sequential=[("movieId", 3706, 5), ("movieId", 3706, 10), ("movieId", 3706, 25), ("movieId", 3706, 50)],
    )

def train_and_evaluate(seq):
    seq.train(train_file, epochs=3, learning_rate=0.0001)
    seq.predict(test_file, predict_file)

def main():
    train_and_evaluate(seq_with_temporal_feats())

if __name__ == "__main__":
    main()
