from thirdai import neural_db as ndb

db = ndb.NeuralDB(num_shards=2, num_models_per_shard=2, extreme_output_dim=500, embedding_dimension=200)

config = ndb.CheckpointConfig(checkpoint_dir="zcheckpoint")

db.insert(
    [ndb.CSV("test.csv", id_column="id", strong_columns=["col", "something"])], 
    epochs=10, 
    train=True, 
    checkpoint_config=config,
)

config = ndb.CheckpointConfig(checkpoint_dir="zcheckpoint", resume_from_checkpoint=True)

db.insert(
    [ndb.CSV("test.csv", id_column="id", strong_columns=["col", "something"])], 
    epochs=3, 
    train=True, 
    checkpoint_config=config,
)