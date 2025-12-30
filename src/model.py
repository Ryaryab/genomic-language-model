# Neural network architecture only. No data, no plotting.

import tensorflow as tf


def build_model(seq_len: int, n_classes: int):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(seq_len,), dtype=tf.int32),
        tf.keras.layers.Embedding(input_dim=64, output_dim=16),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(n_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
