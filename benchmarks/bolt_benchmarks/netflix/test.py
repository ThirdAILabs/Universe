from thirdai import bolt

schema = {
    "user": "user",
    "item": "movie",
    "item_categorical": "release_year",
    "timestamp": "date",
    "target": "rating",
    "item_text": "title",
}

config = bolt.SequentialClassifierConfig(
    task="regression", 
    horizon=1, 
    n_items=17_700, 
    n_users=600_000, 
    n_item_categories=100, 
)

classifier = bolt.SequentialClassifier(schema, config)

classifier.train("/share/data/netflix/netflix_train.csv")
classifier.predict("/share/data/netflix/netflix_test.csv")

