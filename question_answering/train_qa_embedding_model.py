from embedding_model import get_compiled_triplet_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

triplet_network, _ = get_compiled_triplet_model()

queries = np.load("tokenized_queries.npy")
positives = np.load("tokenized_positives.npy")
negatives = np.load("tokenized_negatives.npy")

earlyStopping = EarlyStopping(monitor="val_loss", patience=15, verbose=0, mode="min")
mcp_save = ModelCheckpoint(
    ".mdl_wts.hdf5", save_best_only=True, monitor="val_loss", mode="min"
)
reduce_lr_loss = ReduceLROnPlateau(
    monitor="val_loss", factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode="min"
)

triplet_network.fit(
    [queries, positives, negatives],
    np.zeros((len(queries),)),  # All zeros because we don't use the labels for the loss
    validation_split=0.2,
    batch_size=batch_size,
    epochs=100,
    callbacks=[earlyStopping, mcp_save, reduce_lr_loss],
)

# sentence_embedding_model.save("sentence_embedding_model")
# triplet_network.save("triplet_network")
