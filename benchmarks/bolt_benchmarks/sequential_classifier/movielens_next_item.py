from thirdai import bolt

train_file = "/Users/benitogeordie/Desktop/thirdai_datasets/movielens25m/movielens2019_jan_oct.csv"
test_file = "/Users/benitogeordie/Desktop/thirdai_datasets/movielens25m/movielens2019_nov.csv"
predict_file = "/Users/benitogeordie/Desktop/thirdai_datasets/movielens25m/movielens2019_pred.csv"

def seq_without_temporal_feats():
    return bolt.SequentialClassifier(
        size="small",
        item=("userId", 10601),
        timestamp="timestamp",
        target=("movieId", 41429),
        lookahead=1,  # num days to predict ahead. This is 1 instead of 0 so the trackable categories feature doesnt include the current rating. However I think I should modify it so that it always doesnt include the current category, so if we want to include the current category we should just add a "current" categorical feature. Or autotune it somehow. Figure this out!!!
        lookback=10,  # num days to look back
        metrics=["precision_at_50"]
    )

def seq_with_temporal_feats():
    return bolt.SequentialClassifier(
        size="small",
        item=("userId", 10601),
        timestamp="timestamp",
        target=("movieId", 41429),
        lookahead=0,  # num days to predict ahead. This is 1 instead of 0 so the trackable categories feature doesnt include the current rating. However I think I should modify it so that it always doesnt include the current category, so if we want to include the current category we should just add a "current" categorical feature. Or autotune it somehow. Figure this out!!!
        lookback=14,  # num days to look back
        # Optional:
        # trackable_qty=[""],
        trackable_cat=[("movieId", 41429, 20)],
        metrics=["precision_at_50"]
    )

def train_and_evaluate(seq):
    seq.train(train_file, epochs=5, learning_rate=0.0001,
              validation_filename=test_file)
    return seq.predict(test_file, pred)

def main():
    result_without_temporal_feats = train_and_evaluate(seq_without_temporal_feats())
    result_with_temporal_feats = train_and_evaluate(seq_with_temporal_feats())
    print("Accuracy without temporal feats:", result_without_temporal_feats)
    print("Accuracy with temporal feats:", result_with_temporal_feats)

if __name__ == "__main__":
    main()
