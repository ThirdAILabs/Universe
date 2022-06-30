from thirdai import bolt
from thirdai.dataset import blocks, DataPipeline
import time

pipeline = DataPipeline("/share/data/netflix/netflix_train_10M.csv", [], [blocks.Trend(has_count_col=False, id_col=1, timestamp_col=0, count_col=0, horizon=1, lookback=30)], batch_size=2048, shuffle=True, has_header=True)
start = time.time()
data, labels = pipeline.load_in_memory()
end = time.time()
print("Loaded 10M samples in", end - start, "seconds.")
print(data[4000][10], labels[4000][10])

