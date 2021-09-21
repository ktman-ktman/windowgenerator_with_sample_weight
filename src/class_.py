# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import tensorflow as tf


class WindowGenerator:
    def __init__(
        self,
        input_width: int,
        label_width: int,
        shift: int,
        X_column_l: list,
        Y_column_l: list,
        sample_weight_label_column: str,
    ):
        """[summary]

        Args:
            input_width (int): 特徴量の時系列方向の長さ
            label_width (int): ターゲットの時系列方向の長さ
            shift (int): 何時点先のターゲットを予測するかのシフト量
            training_df (pd.DataFrame): [description]
            validation_df (pd.DataFrame): [description]
            test_df (pd.DataFrame): [description]
            X_column_l (list): 特徴量のカラム名のリスト.
            Y_column_l (list): ターゲットのカラム名のリスト.
            sample_weight_label_column (str): サンプルウェイトのカラム名.
        """
        # Store the raw data.
        self.X_column_l = X_column_l
        self.Y_column_l = Y_column_l
        self.sample_weight_label_column = sample_weight_label_column

        # Work out the window parameters.
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
                f"Label X column name(s): {self.X_column_l}",
                f"Label Y column name(s): {self.Y_column_l}",
                f"Label sample weight column name: {self.sample_weight_label_column}",
            ]
        )

    def _split_window(self, features: np.ndarray) -> tuple:
        """全データをX，Y，sample_weightに分割する

        Args:
            features (np.ndarray): [description]

        Returns:
            tuple: (X, Y, sample_weight)のtuple
        """
        inputs = features[:, self.input_slice, :]
        inputs = tf.stack(
            [inputs[:, :, self.column_indices[name]] for name in self.X_column_l],
            axis=-1,
        )
        labels = features[:, self.labels_slice, :]
        labels = tf.stack(
            [labels[:, :, self.column_indices[name]] for name in self.Y_column_l],
            axis=-1,
        )

        # sample_weights = features[:, self.sample_weight_labels_slice, :]
        sample_weights = features[:, self.labels_slice, :]
        sample_weights = tf.stack(
            [
                sample_weights[
                    :, :, self.column_indices[self.sample_weight_label_column]
                ]
            ],
            axis=-1,
        )

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])
        sample_weights.set_shape([None, self.label_width, None])

        return inputs, labels, sample_weights
        # return inputs, labels

    def make_dataset(
        self, df: pd.DataFrame, batch_size: int, shuffle: bool
    ) -> tf.data.Dataset:
        # Work out the label column indices.
        self.column_indices = {name: i for i, name in enumerate(df.columns)}

        data = np.array(df.values, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=shuffle,
            batch_size=batch_size,
        )

        ds = ds.map(self._split_window)

        return ds
