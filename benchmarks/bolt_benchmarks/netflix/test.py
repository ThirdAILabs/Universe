from thirdai import bolt
import time

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

print("WITH SEQ FEATS")
classifier = bolt.SequentialClassifier(schema, config)

start = time.time()
classifier.train("/share/data/netflix/netflix_train.csv")
end = time.time()
print("TRAINED IN", end - start, "SECONDS")
classifier.predict("/share/data/netflix/netflix_test.csv")

del classifier

print("WITHOUT SEQ FEATS")

classifier = bolt.SequentialClassifier(schema, config, use_sequential_feats=False)

start = time.time()
classifier.train("/share/data/netflix/netflix_train.csv")
end = time.time()
print("TRAINED IN", end - start, "SECONDS")
classifier.predict("/share/data/netflix/netflix_test.csv")

