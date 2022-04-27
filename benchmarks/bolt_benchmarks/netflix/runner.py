from thirdai import bolt, dataset, schema, search

train_path = "/share/data/netflix/date_sorted_data.csv"

# date,user_id,movie_id,rating,movie_title,release_year
date_feats = schema.Date(col=0, timestamp_fmt="%Y-%m-%d", n_years=7)
user_id_feats = schema.OneHotEncoding(col=1, out_dim=2000)
movie_id_feats = schema.OneHotEncoding(col=2, out_dim=20_000)
rating_feats = schema.OneHotEncoding(col=3, out_dim=5)
movie_title_char_feats = schema.CharacterNGram(col=4, k=3, out_dim=15_000)
movie_title_word_feats = schema.WordNGram(col=4, k=1, out_dim=15_000)
release_year = schema.OneHotEncoding(col=5, out_dim=100)

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

movie_watch_rolling_feats = schema.DynamicCounts(
    id_col=2, 
    timestamp_col=0, 
    target_col=-1, 
    window_configs=
        [schema.Window(lag=31 + i, size=1) for i in range(31)]
        + [schema.Window(lag=31 * i, size=7) for i in range(1, 13)]
        + [schema.Window(lag=365 * i, size=7) for i in range(5)], 
    timestamp_fmt="%Y-%m-%d") 

loader = 