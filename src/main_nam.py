#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import click
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from class_nam import WindowGenerator
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

    idx_l = data_df.index.tolist()
    for idx_i in idx_l[-10:]:
        index_loc = data_df.index.get_loc(idx_i)
        train_df = data_df.iloc[:, index_loc - VAL_WIDTH - 1]
        val_df = data_df.iloc[index_loc - VAL_WIDTH - width: index_loc - 1]
        test_df = data_df.iloc[index_loc - width: index_loc]

        # class weight
        classes = np.unique(train_df[target_coln])
        class_weight = comput_class_weight(class_weight="balanced", classes=classes, y=train_df[target_coln])

        # normalize
        scaler = StandardScaler()
        coln_l = [x for x in train_df.columns if x != target_coln]
        train_df.loc[:, coln_l] = scaler.fit_transform(train_df[coln_l])
        val_df.loc[:, coln_l] = scaler.transform(val_df[coln_l])
        test_df.loc[:, coln_l] = scaler.transform(test_df[coln_l])

        window = WindowGenerator(
            input_width=width,
            label_width=1,
            shift=1,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            batch_size=batch_size,
            label_columns=[target_coln],
        )

        set_seed()
        model = get_model()
        history, best_model = compile_and_fit(
            model, window, max_epochs, patience, class_weight=class_weight,
        )
        
        train_loss, train_acc = best_model.evaluate(window.train)
        val_loss, val_acc = best_model.evaluate(window.val)
        prob = best_model.predict_on_batch(test_df.values[np.newaixs, :, :])
        print(idx_i, prob)

        del model
        del window
        del train_df
        del val_df
        del test_df
        tf.keras.backend.clear_session()
        gc.collect()



if __name__ == "__main__":
    main()
