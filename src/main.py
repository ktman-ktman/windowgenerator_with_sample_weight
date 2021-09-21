#!/usr/bin/env python3
#! -*- coding: utf-8 -*-

import datetime
import os

import click
import numpy as np
import pandas as pd

from class_ import WindowGenerator
from model import get_model

os.environ["TF_DETERMINISTIC_OPS"] = "1"


def make_sample_data(target_size: int = 1, feature_size: int = 10) -> pd.DataFrame:
    """サンプルデータを作成
    1990/1/1より後の最も古い金曜日から
    2020/12/31より前の最も新しい金曜日までで
    サンプルデータを作成する

    Args:
        target_size (int, optional): [description]. Defaults to 1.
        feature_size (int, optional): [description]. Defaults to 10.

    Returns:
        pd.DataFrame: [description]
    """

    # make weekly calendar
    date_l = list()
    from_dateymd = 19900101
    to_dateymd = 20201231
    for i in range(7):
        tmp_date = datetime.datetime.strptime(
            str(from_dateymd), "%Y%m%d"
        ) + datetime.timedelta(days=i)
        if tmp_date.weekday() == 4:
            date_l.append(tmp_date)
            break

    else:
        raise Exception("Data error!")

    while True:
        tmp_date = date_l[-1] + datetime.timedelta(days=7)
        if tmp_date >= datetime.datetime.strptime(str(to_dateymd), "%Y%m%d"):
            break

        date_l.append(tmp_date)

    coln_l = ["TARGET"] + [f"F{i+1}" for i in range(feature_size)]
    df = pd.DataFrame(
        np.random.rand(len(date_l), target_size + feature_size),
        columns=coln_l,
        index=date_l,
    )

    # わかりやすいように
    df.loc[:, "TARGET"] = [1 if x > 0.5 else 0 for x in df["TARGET"]]
    df.loc[:, "F1"] = [i for i in range(df.shape[0])]

    return df


def make_sample_weight(sample_df: pd.DataFrame) -> pd.DataFrame:
    if all(sample_df.index != sample_df.index.sort_values()):
        raise Exception("Data error!")

    diff_days = (sample_df.index - sample_df.index[-1]).days

    decay_weight_df = pd.DataFrame(
        diff_days,
        index=sample_df.index,
        columns=["diff_days"],
    )

    half_life_days = 365 * 10
    decay_weight_df = decay_weight_df.assign(
        decay_weight=decay_weight_df["diff_days"].div(half_life_days).abs().rpow(0.5)
    )

    del decay_weight_df["diff_days"]
    return decay_weight_df


@click.command()
@click.argument("input_width", type=int)
@click.argument("batch_size", type=int)
def main(input_width: int, batch_size: int):
    sample_df = make_sample_data()
    training_df = sample_df.iloc[:-53]
    validation_df = sample_df.iloc[-53 : (-1 - input_width + 1)]
    test_df = sample_df.iloc[(-1 - input_width + 1) :]

    # merge with sample weight
    sample_weight_df = make_sample_weight(training_df)
    training_df = pd.merge(
        training_df,
        sample_weight_df,
        how="left",
        left_index=True,
        right_index=True,
    )
    sample_weight_df = make_sample_weight(validation_df)
    validation_df = pd.merge(
        validation_df,
        sample_weight_df,
        how="left",
        left_index=True,
        right_index=True,
    )
    sample_weight_df = make_sample_weight(test_df)
    test_df = pd.merge(
        test_df,
        sample_weight_df,
        how="left",
        left_index=True,
        right_index=True,
    )

    X_column_l = [x for x in sample_df.columns if x.startswith("F")]
    Y_column_l = ["TARGET"]
    sample_weight_label_column = "decay_weight"
    generator = WindowGenerator(
        input_width=input_width,
        label_width=1,
        shift=1,
        X_column_l=X_column_l,
        Y_column_l=Y_column_l,
        sample_weight_label_column=sample_weight_label_column,
    )

    training_ds = generator.make_dataset(
        training_df, batch_size=batch_size, shuffle=True
    )
    validation_ds = generator.make_dataset(
        validation_df, batch_size=batch_size, shuffle=False
    )
    test_input = test_df[X_column_l].values[-input_width:][np.newaxis, :, :]

    input_shape = (input_width, len(X_column_l))
    model = get_model(1, input_shape=input_shape)
    print(model.summary())
    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
    )
    history = model.fit(training_ds, validation_data=validation_ds, epochs=1)
    print(model.predict(test_input))


if __name__ == "__main__":
    main()
