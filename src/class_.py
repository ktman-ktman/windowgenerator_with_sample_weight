# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf


class WindowGenerator:
    def __init__(
        self,
        input_width: int,
        label_width: int,
        shift: int,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        label_columns: list,
        use_label_flag: bool = False,
        batch_size: int = 128,
    ):
        """_summary_

        Args:
            input_width (int): 予測に利用するステップ数．過去何ステップのデータを使って予測するか?
            label_width (int): 予測されるステップ数．翌ステップのみ予測なら1．
            shift (int): 何個先のステップを予測するか?翌ステップのみ予測なら1．
            train_df (pd.DataFrame): training data
            val_df (pd.DataFrame): validation data
            test_df (pd.DataFrame): test data
            label_columns (list): targetとなるカラム名
            use_label_flag (bool, optional): label_columns自体も特徴量に利用するかどうか. Defaults to False.
            batch_size (int, optional): batch size. Defaults to 128.

        Raises:
            Exception: _description_
            Exception: _description_
            Exception: _description_
        """

        # Store the raw data.
        self.train_df = train_df.copy()
        self.val_df = val_df.copy()
        self.test_df = test_df.copy()
        self.batch_size = batch_size

        # データチェック
        ## 特徴量は全て同じ順番であることが求められる．
        ## 次のステップで仮定．
        if not all(train_df.columns == val_df.columns):
            raise Exception("Validation data columns don't match with training data.")

        elif not all(train_df.columns == test_df.columns):
            raise Exception("Test data columns don't match with training data.")

        # label(target)の情報(列番号)を取得
        ## self.label_columns: list
        ##      label_columnsと同等
        ## self.label_columns_indices: dict
        ##      予測対象の情報
        ##      予測対象の名前をkey，その列番号をvalueとする辞書．
        ## self.columns_indices: dict | None
        ##      特徴量の情報
        ##      特徴量の名前をkey，その列番号をvalueとする辞書．
        ##      use_label_flagがTrueの場合はlabel_columnsのデータは特徴量として利用しない
        self.label_columns = label_columns
        self.use_label_flag = use_label_flag
        # label_columnsが指定されていたらその列番号を記憶
        self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
        if self.use_label_flag:
            # label_columnsも特徴量に利用する場合
            self.column_indices = {name: i for i, name in enumerate(train_df.columns)}
        else:
            # label_columnsは特徴量に利用しない場合
            self.column_indices = {
                name: i
                for i, name in enumerate(train_df.columns)
                if name not in label_columns
            }
            if len(self.column_indices) == 0:
                # 与えられたデータが全て予測対象なら特徴量がなくなるのでエラー
                raise Exception("There is no features data.")

        # window関連のパラメータ設定
        ## self.input_width: int
        ##      input_widthと同等
        ## self.label_width: int
        ##      label_widthと同等
        ## self.label_width: shift
        ##      shiftと同等
        ## self.total_window_size: int
        ##      input_width+shiftで計算される全データ長
        ## self.input_slice: slice
        ##      self.total_window_sizeからinput_widthを抽出するためのslice
        ## self.input_indices: np.ndarray
        ##      self.total_window_sizeからinput_widthによって抽出されたindex
        ##      入力データの場所を指定するために利用
        ## self.label_start: int
        ##      self.total_window_sizeからlabelt_widthが開始する場所
        ## self.label_slice: slice
        ##      self.total_window_sizeからlabel_widthを抽出するためのslice
        ## self.label_indices: np.ndarray
        ##      self.total_window_sizeからlabel_widthによって抽出されたindex
        ##      予測データの場所を指定するために利用
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return "\n".join(
            [
                f"Total window size: {self.total_window_size}",
                f"Input indices: {self.input_indices}",
                f"Label indices: {self.label_indices}",
                f"Label column name(s): {self.label_columns}",
            ]
        )

    def split_window(self, features: tf.Tensor) -> tuple:
        """windowに結合されたデータを
        入力とtagertに分割する．

        Args:
            features (tf.Tensor): 結合されたデータ

        Returns:
            tuple:
                inputs: 入力データ
                labels: targetデータ
        """

        # featuresをinputsとlabelsに分解
        ## features: (batch, total_window_size, features)
        ## inputs: (batch, input_width, features)
        ## labels: (batch, label_width, features)
        ## ここでself.labels_sliceはshift分を考慮されている
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]

        # inputsから特徴量のみ抽出
        inputs = tf.stack(
            [
                inputs[:, :, self.column_indices[name]]
                for name in self.column_indices.keys()
            ],
            axis=-1,
        )

        # labelsから特徴量のみ抽出
        labels = tf.stack(
            [
                labels[:, :, self.label_columns_indices[name]]
                for name in self.label_columns_indices.keys()
            ],
            axis=-1,
        )

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def plot(self, model=None, plot_col="p (mbar)", max_subplots=3):
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n + 1)
            plt.ylabel(f"{plot_col} [normed]")
            plt.plot(
                self.input_indices,
                inputs[n, :, plot_col_index],
                label="Inputs",
                marker=".",
                zorder=-10,
            )

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(
                self.label_indices,
                labels[n, :, label_col_index],
                edgecolors="k",
                label="Labels",
                c="#2ca02c",
                s=64,
            )
            if model is not None:
                predictions = model(inputs)
                plt.scatter(
                    self.label_indices,
                    predictions[n, :, label_col_index],
                    marker="X",
                    edgecolors="k",
                    label="Predictions",
                    c="#ff7f0e",
                    s=64,
                )

            if n == 0:
                plt.legend()

        plt.xlabel("Time [h]")

    def make_dataset(self, data, is_shuffle: bool):
        """tf.datasetを作成．
        targetはtimeseries_dataset_from_arrayでは設定しない．
        split_windowで作成するため．

        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=is_shuffle,
            batch_size=self.batch_size,
        )

        ds = ds.map(self.split_window)

        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df, is_shuffle=True)

    @property
    def val(self):
        return self.make_dataset(self.val_df, is_shuffle=False)

    @property
    def test(self):
        return self.make_dataset(self.test_df, is_shuffle=False)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, "_example", None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result
