# -*- coding: utf-8 -*-

import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf


def set_seed(seed=200):
    tf.random.set_seed(seed)

    # optional
    # for numpy.random
    np.random.seed(seed)
    # for built-in random
    random.seed(seed)
    # for hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def make_sample_data() -> pd.DataFrame:
    """サンプルデータを作成
    tutorialのclimateデータにする

    Returns:
        pd.DataFrame: _description_
    """
    zip_path = tf.keras.utils.get_file(
        origin="https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip",
        fname="jena_climate_2009_2016.csv.zip",
        extract=True,
    )
    csv_path, _ = os.path.splitext(zip_path)
    df = pd.read_csv(csv_path, na_values=-9999)
    df.dropna(inplace=True)

    df.loc[:, "Date Time"] = pd.to_datetime(df["Date Time"], format="%d.%m.%Y %H:%M:%S")
    df.set_index("Date Time", inplace=True)
    return df


def compile_and_fit(
    model: tf.keras.Model, window, max_epochs: int = 10, patience: int = 2
):
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=patience, mode="auto"
    )

    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )

    history = model.fit(
        window.train,
        epochs=max_epochs,
        validation_data=window.val,
        callbacks=[early_stopping],
    )
    return history
