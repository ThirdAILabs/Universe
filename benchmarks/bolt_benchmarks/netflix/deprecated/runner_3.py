from thirdai import bolt, dataset, schema, search
import time

file_path = "/share/data/netflix/date_sorted_data.csv"

print(f"File path is {file_path}")

input_blocks = []
label_blocks = []

# date,user_id,movie_id,rating,movie_title,release_year
date_feats = schema.Date(col=0, timestamp_fmt="%Y-%m-%d", n_years=7)
input_blocks.append(date_feats)
user_id_feats = schema.OneHotEncoding(col=1, out_dim=480_189)
input_blocks.append(user_id_feats)
movie_id_feats = schema.OneHotEncoding(col=2, out_dim=17_770)
input_blocks.append(movie_id_feats)
rating_feats = schema.Number(col=3)
label_blocks.append(rating_feats)
movie_title_char_feats = schema.CharacterNGram(col=4, k=3, out_dim=15_000)
input_blocks.append(movie_title_char_feats)
movie_title_word_feats = schema.WordNGram(col=4, k=1, out_dim=15_000)
input_blocks.append(movie_title_word_feats)
release_year_feats = schema.OneHotEncoding(col=5, out_dim=100)
input_blocks.append(release_year_feats)

loader = schema.DataLoader(
    input_block_configs=input_blocks,
    label_block_configs=label_blocks,
    batch_size=2048
)

start = time.time()
loader.read_csv(file_path, delimiter=",")
end = time.time()

start = time.time()
print("Started exporting train data at", start)
train_data = loader.export_dataset(max_export=98074929, shuffle=True)
end = time.time()
print("Finished exporting train data at", end)
print("That took", end - start, "seconds.")

start = time.time()
print("Started exporting test data at", start)
test_data = loader.export_dataset(shuffle=False)
end = time.time()
print("Finished exporting test data at", end)
print("That took", end - start, "seconds.")

layers = [
    bolt.FullyConnected(
        dim=5000,
        load_factor=0.02,
        activation_function=bolt.ActivationFunctions.ReLU
    ),
    bolt.FullyConnected(dim=1, activation_function=bolt.ActivationFunctions.Linear),
]
network = bolt.Network(layers=layers, input_dim=loader.input_dim())

for _ in range(20):
    network.train(
        train_data,
        bolt.MeanSquaredError(),
        0.0001,
        1,
        rehash=6400,
        rebuild=128000,
        metrics=["root_mean_squared_error"],
        verbose=True
    )
    network.predict(
        test_data, metrics=["root_mean_squared_error"], verbose=True
    )
