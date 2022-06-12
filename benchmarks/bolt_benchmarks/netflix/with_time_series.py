from thirdai.dataset import blocks
from model_runner import run_experiment

# FILE FORMAT
# date,user_id,movie_id,rating,movie_title,release_year
windows = [
    blocks.Window(lag=1, size=1),
    blocks.Window(lag=2, size=1),
    blocks.Window(lag=3, size=1),
    blocks.Window(lag=4, size=1),
    blocks.Window(lag=5, size=1),
    blocks.Window(lag=10, size=5),
    blocks.Window(lag=20, size=10),
    blocks.Window(lag=30, size=10),
]

time_series_index_config = blocks.DynamicCountsConfig(
    max_range=10, 
    lifetime_in_days=35, 
    n_rows=5, 
    range_pow=22,
)

input_blocks = [
    blocks.Date(col=0), # date column
    blocks.Categorical(col=1, dim=480_189), # user id column
    blocks.Categorical(col=2, dim=17_770), # movie id column
    blocks.Text(col=4, dim=100_000), # movie title column
    blocks.Categorical(col=5, dim=100), # release year column
    blocks.CountHistory(
        has_count_col=False, timestamp_col=0, id_col=1, # user watch count history
        count_col=0, windows=windows, 
        index_config=time_series_index_config), 
    blocks.CountHistory(
        has_count_col=False, timestamp_col=0, id_col=2, # movie view count history
        count_col=0, windows=windows, 
        index_config=time_series_index_config), 
]

label_blocks = [
    blocks.Continuous(col=3) # rating column
]

run_experiment(input_blocks, label_blocks)

