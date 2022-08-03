from thirdai import bolt

def seq_without_temporal_feats():
    return bolt.SequentialClassifier(
        size="medium",
        item=("user", 480189),
        timestamp="date",
        target=("movie", 17770),
        lookahead=1,  # num days to predict ahead. This is 1 instead of 0 so the trackable categories feature doesnt include the current rating. However I think I should modify it so that it always doesnt include the current category, so if we want to include the current category we should just add a "current" categorical feature. Or autotune it somehow. Figure this out!!!
        lookback=10,  # num days to look back
        # Optional:
        period=1,  # expected num days between each record; period for clubbing data points together
    )

def seq_with_temporal_feats():
    return bolt.SequentialClassifier(
        size="medium",
        item=("user", 480189),
        timestamp="date",
        target=("movie", 17770),
        lookahead=0,  # num days to predict ahead. This is 1 instead of 0 so the trackable categories feature doesnt include the current rating. However I think I should modify it so that it always doesnt include the current category, so if we want to include the current category we should just add a "current" categorical feature. Or autotune it somehow. Figure this out!!!
        lookback=14,  # num days to look back
        # Optional:
        period=1,  # expected num days between each record; period for clubbing data points together
        trackable_qty=[""],
        trackable_cat=[("movie", 17770, 20)],
        metrics=["categorical_accuracy", "precision_at_10", "precision_at_25", "precision_at_50", "precision_at_100"]
    )

def train_and_evaluate(seq, suffix):
    seq.train("/share/data/netflix/netflix_train.csv-binary-class", epochs=5, learning_rate=0.0001,
              validation_filename="/share/data/netflix/netflix_test.csv-binary-class")
    return seq.predict("/share/data/netflix/netflix_test.csv-binary-class", "netflix_predictions_next_item_" + suffix + ".txt")

def main():
    result_without_temporal_feats = train_and_evaluate(seq_without_temporal_feats(), "no_temporal")
    result_with_temporal_feats = train_and_evaluate(seq_with_temporal_feats(), "temporal")
    print("Accuracy without temporal feats:", result_without_temporal_feats)
    print("Accuracy with temporal feats:", result_with_temporal_feats)

if __name__ == "__main__":
    main()
