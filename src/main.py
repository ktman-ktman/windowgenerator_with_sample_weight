#!/usr/bin/env python3
#! -*- coding: utf-8 -*-

import datetime

import numpy as np
import pandas as pd

from class_ import WindowGenerator


def make_sample_data(target_size: int = 1, feature_size: int = 2) -> pd.DataFrame:
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
    df.loc[:, "TARGET"] = [i for i in range(df.shape[0])]
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


def main():
    sample_df = make_sample_data()

    # make sample weight
    sample_weight_df = make_sample_weight(sample_df)
    sample_df = pd.merge(
        sample_df,
        sample_weight_df,
        how="left",
        left_index=True,
        right_index=True,
    )
    training_df = sample_df.iloc[:-53]
    validation_df = sample_df.iloc[-53:-1]
    test_df = sample_df.iloc[-1:]

    generator = WindowGenerator(
        input_width=4,
        label_width=1,
        shift=1,
        training_df=training_df,
        validation_df=validation_df,
        test_df=test_df,
        X_column_l=[x for x in sample_df.columns if x.startswith("F")],
        Y_column_l=["TARGET"],
        # exclude_target=True,
        sample_weight_label_column="decay_weight",
    )

    training_ds = generator.training
    labels, inputs, sample_weights = next(iter(training_ds.take(1)))
    print(labels)
    print(inputs)
    print(sample_weights)


if __name__ == "__main__":
    main()
