from tensorflow import keras
from tensorflow.keras.layers import Embedding, Dense, Lambda, Input, BatchNormalization
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np

queries = np.load("tokenized_queries.npy")
positives = np.load("tokenized_positives.npy")
negatives = np.load("tokenized_negatives.npy")

embedding_dim = 128
vocab_size = 30522
batch_size = 1024


def get_embedding_model():
    model = keras.Sequential()
    # Embedding layer
    model.add(Embedding(vocab_size, embedding_dim, input_length=None))
    # Average tokens
    model.add(keras.layers.Lambda(lambda x: keras.backend.mean(x, axis=1)))
    # Single MLP layer
    model.add(Dense(embedding_dim, activation="relu"))

    return model


# Used multiple times by the triplet model, and can later save as the embedding model
sentence_embedding_model = get_embedding_model()


def get_triplet_model():

    # See https://zhangruochi.com/Create-a-Siamese-Network-with-Triplet-Loss-in-Keras/2020/08/11/
    input_query = tf.keras.layers.Input(shape=(None,))
    input_positive_passage = tf.keras.layers.Input(shape=(None,))
    input_negative_passage = tf.keras.layers.Input(shape=(None,))

    embedding_model_query = sentence_embedding_model(input_query)
    embedding_model_positive_passage = sentence_embedding_model(input_positive_passage)
    embedding_model_negative_passage = sentence_embedding_model(input_negative_passage)

    output = tf.keras.layers.concatenate(
        [
            embedding_model_query,
            embedding_model_positive_passage,
            embedding_model_negative_passage,
        ],
        axis=1,
    )

    return tf.keras.models.Model(
        [input_query, input_positive_passage, input_negative_passage], output
    )


alpha = 0.5


def triplet_loss(y_true, y_pred):
    query_embedding, positive_embedding, negative_embedding = (
        y_pred[:, :embedding_dim],
        y_pred[:, embedding_dim : 2 * embedding_dim],
        y_pred[:, 2 * embedding_dim :],
    )
    query_embedding = tf.math.l2_normalize(query_embedding, axis=1)
    positive_embedding = tf.math.l2_normalize(positive_embedding, axis=1)
    negative_embedding = tf.math.l2_normalize(negative_embedding, axis=1)
    positive_dist = 1 - tf.keras.backend.batch_dot(
        query_embedding, positive_embedding, axes=1
    )
    negative_dist = 1 - tf.keras.backend.batch_dot(
        query_embedding, negative_embedding, axes=1
    )
    return tf.maximum(positive_dist - negative_dist + alpha, 0)


def get_compiled_triplet_model():
    triplet_model = get_triplet_model()
    triplet_model.compile(loss=triplet_loss, optimizer="adam")
    return triplet_model


triplet_network = get_compiled_triplet_model()

triplet_network.fit(
    [queries, positives, negatives],
    np.zeros((len(queries),)),  # All zeros because we don't use the labels for the loss
    validation_split=0.2,
    batch_size=batch_size,
    epochs=5,
)

sentence_embedding_model.save("sentence_embedding_model")
triplet_network.save("triplet_network")
