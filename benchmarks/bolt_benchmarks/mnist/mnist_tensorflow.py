import mlflow.tensorflow
import tensorflow as tf
import tensorflow_datasets as tfds
from mlflow_logger import ExperimentLogger

# TODO(vihan): Move this into a separate file once we have more TF scripts
class ValidationMetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, mlflow_logger):
        self.logger = mlflow_logger

    def on_epoch_end(self, epoch, logs=None):
        self.logger.log_epoch(logs.get("val_sparse_categorical_accuracy", 0.0))


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255.0, label


def train(ds_train, ds_test, mlflow_logger):
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(10),
        ]
    )
    validation_callback = ValidationMetricsCallback(mlflow_logger)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    mlflow.tensorflow.autolog(every_n_iter=1)
    mlflow_logger.log_start_training()
    model.fit(
        ds_train,
        epochs=6,
        validation_data=ds_test,
        callbacks=[validation_callback],
    )


def main():
    (ds_train, ds_test), ds_info = tfds.load(
        "mnist",
        split=["train", "test"],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(128)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    with ExperimentLogger(
        experiment_name="MNIST Benchmark",
        dataset="mnist",
        algorithm="feedforward",
        framework="TensorFlow",
    ) as mlflow_logger:
        train(ds_train, ds_test, mlflow_logger)


if __name__ == "__main__":
    main()
