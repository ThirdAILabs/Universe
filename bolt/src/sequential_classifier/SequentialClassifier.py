seq = SequentialClassifier(
    schema = {
        "user": user_col,
        "item": item_col,
        "timestamp": timestamp_col,
        "target": target_col,
        "trackable_quantity": trackable_qty_col,
        "categorical": categorical_col,
        "text": text_col,
    }, 
    config = SequentialClassifierConfig(
        model_size = "small",
        n_target_classes = 4,
        horizon = 0, 
        n_items = 50,
        n_users = 50,
        n_categories = 50,
        user_graph = graph,
        max_user_neighbors = 10,
        item_graph = graph,
        max_item_neighbors = 10,
    )
)

seq = SequentialClassifier(
    size="small",
    item=("item_col", n_items, graph, max_item_neighbors),
    timestamp="timestamp_col",
    target=("target_col", n_targets),
    horizon=7, # 7 days
    lookback=30, # 30 days
    # Optional:
    period=7, # 7 days
    text=["text_col_1", "text_col_2", "text_col_3"],
    categorical=[
        ("cat_col_1", n_cat_1),
        ("cat_col_2", n_cat_2),
        ("cat_col_3", n_cat_3),
    ],
    trackable_qty=["trackable_qty_1", "trackable_qty_2", "trackable_qty_3"],
    metadata=metadata_dict
)

