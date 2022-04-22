from base_experiment import run_experiment
from thirdai import schema

discard_filename = "uid_discard.csv"
train_filename = "uid_train.csv"
test_filename = "uid_test.csv"

# Data columns:
# ASIN\tSHIP_WINDOW_END\tVENDOR\tSUBMITTED_UNITS\tSUBMITTED_UNITS_30D\tPRODUCT_TITLE\tCATEGORY\tBRAND\tSUB_CATEGORY
# Delimiter is \t.

schema_feats = []
params = {}

# ASIN: contiguous integers from 0 to 8663.
asin_feats = schema.OneHotEncodingBlock(col=0, out_dim=8664)
schema_feats.append(asin_feats)
params["asin"] = "one_hot"

# SHIP_WINDOW_END: date in yyyy-mm-dd format.
date_feats = schema.DateBlock(col=1, timestamp_fmt="%Y-%m-%d", n_years=3)
schema_feats.append(date_feats)
params["date"] = "standard_date"

# VENDOR: contiguous integers from 0 to 57.
vendor_feats = schema.CharacterNGramBlock(col=2, k=5, out_dim=30)
schema_feats.append(vendor_feats)
params["vendor"] = "one_hot"

# SUBMITTED_UNITS: time series. save for last. col 3.

# SUBMITTED_UNITS_30D: number. label.
label_block = schema.NumericalLabelBlock(col=4)
schema_feats.append(label_block)
params["submitted_units_30d"] = "num_label"

# PRODUCT_TITLE: text with all \t's removed.
product_title_feats_1 = schema.CharacterNGramBlock(col=5, k=3, out_dim=15000)
product_title_feats_2 = schema.WordNGramBlock(col=5, k=1, out_dim=15000)
schema_feats.append(product_title_feats_1)
schema_feats.append(product_title_feats_2)
params["product_title"] = "char_trigram:15_000,word_unigram:15_000"

# CATEGORY: contiguous integers from 0 to 90.
category_feats = schema.OneHotEncodingBlock(col=6, out_dim=91)
schema_feats.append(category_feats)
params["category"] = "one_hot"

# BRAND: contiguous integers from 0 to 155.
brand_feats = schema.OneHotEncodingBlock(col=7, out_dim=156)
schema_feats.append(brand_feats)
params["brand"] = "one_hot"

# SUB_CATEGORY: contiguous integers from 0 to 262.
sub_category_feats = schema.OneHotEncodingBlock(col=8, out_dim=263)
schema_feats.append(sub_category_feats)
params["sub_category"] = "one_hot"

# Set up dynamic counts block for UNITS

# The input features are lagged windowed counts.
min_lag = 92
# Lag is at least min_lag since we are predicting min_lag days ahead (3 months).
input_feature_window_configs = [schema.Window(lag=min_lag + i, size=1) for i in range(92)]

ordered_items_feats = schema.DynamicCountsBlock(
    id_col=0, # ASIN column
    timestamp_col=1, # SHIP_WINDOW_END column 
    target_col=3, # SUBMITTED_UNITS column
    input_window_configs=input_feature_window_configs, 
    label_window_configs= [], # no label config because we are using numerical label block for that.
    timestamp_fmt="%Y-%m-%d")

schema_feats.append(ordered_items_feats)
params["submitted_units"] = "daily_counts_last_92d"

# Assemble schema!
batch_size = 256
loader = schema.DataLoader(schema=schema_feats, batch_size=batch_size)
input_dim = loader.input_feat_dim()

def get_train_and_test():
    loader.read_csv(discard_filename, delimiter='\t')
    discard = loader.export_dataset() 
    del discard # discard this. We only read it for the time series index, not for input.

    loader.read_csv(train_filename, delimiter='\t')
    train_dataset = loader.export_dataset()
    
    loader.read_csv(test_filename, delimiter='\t')
    test_dataset = loader.export_dataset()

    return train_dataset, test_dataset


run_experiment(
    dataset="CommerceIQ", 
    hidden_dim=5_000, 
    sparsity=0.02,
    get_train_and_test_fn=get_train_and_test,
    input_dim=input_dim,
    epochs=1000,
    params=params,
    )

