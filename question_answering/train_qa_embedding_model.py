from embedding_model import get_compiled_triplet_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np

triplet_network, _ = get_compiled_triplet_model(learning_rate=0.01)

queries = np.load("tokenized_queries.npy")
passages_1 = np.load("tokenized_passages_1.npy")
passages_2 = np.load("tokenized_passages_2.npy")
labels = np.load("labels.npy")

earlyStopping = EarlyStopping(monitor="val_loss", patience=15, verbose=0, mode="min")
mcp_save = ModelCheckpoint(
    ".mdl_wts.hdf5", save_best_only=True, monitor="val_loss", mode="min"
)
reduce_lr_loss = ReduceLROnPlateau(
    monitor="val_loss", factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode="min"
)

batch_size = 1024
triplet_network.fit(
    [queries, passages_1, passages_2],
    labels,
    validation_split=0.2,
    batch_size=batch_size,
    epochs=100,
    callbacks=[earlyStopping, mcp_save, reduce_lr_loss],
)
