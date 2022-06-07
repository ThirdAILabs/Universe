from tensorflow import keras
from tensorflow.keras.layers import (
    Embedding,
    Dense,
    Lambda,
    Input,
    Dot,
    BatchNormalization,
    Concatenate,
)
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np

alpha = 0.5
embedding_dim = 256
vocab_size = 30522


def get_embedding_model():
    model = keras.Sequential()
    # Embedding layer
    model.add(Embedding(vocab_size, embedding_dim, input_length=None))
    # Average tokens
    model.add(keras.layers.Lambda(lambda x: keras.backend.mean(x, axis=1)))
    # Single MLP layer
    model.add(Dense(embedding_dim, activation="relu"))

    return model


def get_triplet_and_sub_model():

    # Used multiple times by the triplet model
    sentence_embedding_model = get_embedding_model()

    # See https://zhangruochi.com/Create-a-Siamese-Network-with-Triplet-Loss-in-Keras/2020/08/11/
    input_query = Input(shape=(None,))
    input_passage_1 = Input(shape=(None,))
    input_passage_2 = Input(shape=(None,))

    embedding_model_query = sentence_embedding_model(input_query)
    embedding_model_passage_1 = sentence_embedding_model(input_passage_1)
    embedding_model_passage_2 = sentence_embedding_model(input_passage_2)

    sim_1 = Dot(axes=1, normalize=True)(
        [embedding_model_query, embedding_model_passage_1]
    )
    sim_2 = Dot(axes=1, normalize=True)(
        [embedding_model_query, embedding_model_passage_2]
    )

    concatenated_sims = Concatenate(axis=1)([sim_1, sim_2])

    output = Dense(1, activation="relu")(concatenated_sims)

    return (
        tf.keras.models.Model([input_query, input_passage_1, input_passage_2], output),
        sentence_embedding_model,
    )


def get_compiled_triplet_model(learning_rate):
    triplet_model, sentence_embedding_model = get_triplet_and_sub_model()
    triplet_model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=[tf.keras.metrics.BinaryAccuracy()],
    )
    return triplet_model, sentence_embedding_model
