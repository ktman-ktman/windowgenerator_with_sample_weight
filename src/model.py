# -*- coding: utf-8 -*-

import tensorflow as tf


def get_model(model_number: int, input_shape: tuple) -> tf.keras.Model:
    """モデルを作成する

    Args:
        model_number (int): _description_
        input_shape (tuple): _description_

    Raises:
        Exception: _description_

    Returns:
        tf.keras.Model: _description_
    """
    if model_number not in [1]:
        raise Exception("Data error!")

    if model_number == 1:
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.LSTM(16, input_shape=input_shape))
        model.add(tf.keras.layers.Dense(16))
        model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    return model
