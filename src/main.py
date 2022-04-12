#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import click
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from class_ import WindowGenerator
from functions_ import compile_and_fit, make_sample_data
from model import get_model

os.environ["TF_DETERMINISTIC_OPS"] = "1"

import os
import random

import numpy as np
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


@click.command()
@click.argument("input_width", type=int)
@click.argument("batch_size", type=int)
def main(input_width: int, batch_size: int):
    target_coln = "p (mbar)"
    data_df = make_sample_data()
    data_df = data_df.iloc[:10000]

    # targetを0,1に変更
    ## class weightに対応するため少しゆがめる
    data_df.loc[:, target_coln] = [
        1 if x >= data_df[target_coln].mean() - data_df[target_coln].std() else 0
        for x in data_df[target_coln]
    ]

    # データを分割する
    train_df = data_df.iloc[: int(data_df.shape[0] * 0.6)].copy()
    val_df = data_df.iloc[
        int(data_df.shape[0] * 0.6) : int(data_df.shape[0] * 0.9)
    ].copy()
    test_df = data_df.iloc[int(data_df.shape[0] * 0.9) :].copy(9)

    # normalize
    X_column_l = [x for x in data_df.columns if x != target_coln]
    Y_column_l = [target_coln]

    scaler = StandardScaler()
    train_df.loc[:, X_column_l] = scaler.fit_transform(train_df[X_column_l])
    val_df.loc[:, X_column_l] = scaler.transform(val_df[X_column_l])
    test_df.loc[:, X_column_l] = scaler.transform(test_df[X_column_l])

    generator = WindowGenerator(
        input_width=input_width,
        label_width=1,
        shift=1,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        label_columns=Y_column_l,
        batch_size=batch_size,
    )

    input_shape = (input_width, len(X_column_l))
    set_seed()
    model = get_model(1, input_shape=input_shape)
    print(model.summary())
    compile_and_fit(model, generator)
    model.evaluate(generator.test)


if __name__ == "__main__":
    main()
