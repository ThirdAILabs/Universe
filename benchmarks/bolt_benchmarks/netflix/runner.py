from thirdai import bolt, dataset, schema, search
import time

train_path = "/share/data/netflix/date_sorted_data.csv"

input_blocks = []
label_blocks = []

# date,user_id,movie_id,rating,movie_title,release_year
date_feats = schema.Date(col=0, timestamp_fmt="%Y-%m-%d", n_years=7)
input_blocks.append(date_feats)
user_id_feats = schema.OneHotEncoding(col=1, out_dim=480_189)
input_blocks.append(user_id_feats)
movie_id_feats = schema.OneHotEncoding(col=2, out_dim=17_770)
input_blocks.append(movie_id_feats)
rating_feats = schema.OneHotEncoding(col=3, out_dim=5)
label_blocks.append(rating_feats)
movie_title_char_feats = schema.CharacterNGram(col=4, k=3, out_dim=15_000)
input_blocks.append(movie_title_char_feats)
movie_title_word_feats = schema.WordNGram(col=4, k=1, out_dim=15_000)
input_blocks.append(movie_title_word_feats)
release_year_feats = schema.OneHotEncoding(col=5, out_dim=100)
input_blocks.append(release_year_feats)

# target_col=-1 because there is no target column; 
# we just consider each record to be one watching session.
user_watch_rolling_feats = schema.DynamicCounts(
    id_col=1, 
    timestamp_col=0, 
    target_col=-1, 
    window_configs=
        [schema.Window(lag=31 + i, size=1) for i in range(31)]
        + [schema.Window(lag=31 * i, size=7) for i in range(1, 13)]
        + [schema.Window(lag=365 * i, size=7) for i in range(5)], 
    timestamp_fmt="%Y-%m-%d") 
input_blocks.append(user_watch_rolling_feats)

movie_watch_rolling_feats = schema.DynamicCounts(
    id_col=2, 
    timestamp_col=0, 
    target_col=-1, 
    window_configs=
        [schema.Window(lag=31 + i, size=1) for i in range(31)]
        + [schema.Window(lag=31 * i, size=7) for i in range(1, 13)]
        + [schema.Window(lag=365 * i, size=7) for i in range(5)], 
    timestamp_fmt="%Y-%m-%d") 
input_blocks.append(movie_watch_rolling_feats)

loader = schema.DataLoader(
    input_block_configs=input_blocks,
    label_block_configs=label_blocks,
    batch_size=2048
)

start = time.time()
print("Started reading dataset at", start)
loader.read_csv('/share/data/netflix/date_sorted_data.csv', delimiter=",")
end = time.time()
print("Finished reading dataset at", end)
print("That took", end - start, "seconds.")

start = time.time()
print("Started exporting train data at", start)
# train_data = loader.export_dataset(max_export=98074929, shuffle=False)
train_data = loader.export_dataset(max_export=10000, shuffle=False)
end = time.time()
print("Finished exporting train data at", end)
print("That took", end - start, "seconds.")

start = time.time()
print("Started exporting test data at", start)
# test_data = loader.export_dataset(shuffle=False)
test_data = loader.export_dataset(max_export=1000, shuffle=False)
end = time.time()
print("Finished exporting test data at", end)
print("That took", end - start, "seconds.")

layers = [
    bolt.LayerConfig(
        dim=5000,
        load_factor=0.02,
        activation_function=bolt.ActivationFunctions.ReLU
    ),
    bolt.LayerConfig(dim=2, activation_function=bolt.ActivationFunctions.Softmax),
]
network = bolt.Network(layers=layers, input_dim=loader.input_dim())

for _ in range(10):
    network.train(
        train_data,
        bolt.CategoricalCrossEntropyLoss(),
        0.0001,
        1,
        rehash=6400,
        rebuild=128000,
        verbose=True
    )
    network.predict(
        test_data, metrics=["categorical_accuracy"], verbose=True
    )