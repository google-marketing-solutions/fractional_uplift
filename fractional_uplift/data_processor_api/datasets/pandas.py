# Copyright 2024 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The new API for data processing.

For now this is experimental. It is not being used in fractional uplift.
"""

from typing import Any

import numpy as np
import pandas as pd

from fractional_uplift.data_processor_api.datasets import base


class PandasDataset(base.Dataset):
  """A dataset that is backed by pandas.

  The dataset can perform a variety of data processing actions, and can export
  the dataset to a varity of formats, but it stores and manipulates the data
  with pandas.
  """

  def __init__(self, data: pd.DataFrame):
    self.data = data

  def as_pd_dataframe(self) -> pd.DataFrame:
    """Returns the dataset as a pandas dataframe."""
    return self.data

  def column_is_not_negative(self, column_name: str) -> bool:
    """Returns true if the column is not negative for all rows."""
    return np.all(self.data[column_name].values >= 0.0)

  def column_exists(self, column_name: str) -> bool:
    """Returns true if the column exists in the dataset."""
    return column_name in self.data.columns

  def copy(self) -> "PandasDataset":
    """Returns a copy of the dataset."""
    return self.__class__(data=self.data.copy())

  def set_column_from_constant(
      self, column_name: str, column_value: Any
  ) -> "PandasDataset":
    """Creates a column with the given name and constant value."""
    self.data[column_name] = column_value
    return self

  def set_column_from_numpy_array(
      self, column_name: str, column_values: np.ndarray
  ) -> "PandasDataset":
    """Creates a column from a numpy array."""
    self.data[column_name] = column_values
    return self

  def filter(self, mask_column: str) -> "PandasDataset":
    """Filters the dataset to only rows where the mask_column is true."""
    self.data = self.data[self.data[mask_column]]
    return self

  def append_rows(self, data: "PandasDataset") -> "PandasDataset":
    """Appends the rows of the given dataset to the end of this dataset.

    This ignores the index of the pandas dataframe, so the index gets reset.

    Args:
      data: The dataset to append to this dataset.

    Returns:
      The concatenated dataset.

    Raises:
      ValueError: If the columns don't match between the two datasets.
    """
    if set(self.get_columns()) != set(data.get_columns()):
      raise ValueError(
          "The columns don't match between the two datasets:"
          f" {self.get_columns() = } != {data.get_columns() = }"
      )

    self.data = pd.concat(
        [self.data, data.as_pd_dataframe()[self.get_columns()]],
        ignore_index=True,
    )
    return self

  def set_column_from_subtraction(
      self,
      *,
      output_column_name: str,
      minuend_column_name: str,
      subtrahend_column_name: str,
  ) -> "PandasDataset":
    """Creates a column as the subtraction of the subtrahend from the minuend."""
    self.data[output_column_name] = (
        self.data[minuend_column_name] - self.data[subtrahend_column_name]
    )
    return self

  def set_column_from_addition(
      self, output_column_name: str, *addition_column_names: str
  ) -> "PandasDataset":
    """Creates a column that is the sum of the addition column names."""
    raise NotImplementedError()

  def set_column_from_division(
      self,
      output_column_name: str,
      numerator_column_name: str,
      denominator_column_name: str,
  ) -> "PandasDataset":
    """Creates a column as the division of the numerator and denominator."""
    raise NotImplementedError()

  def set_column_from_multiplication(
      self, output_column_name: str, *multiply_column_names: str
  ) -> "PandasDataset":
    """Creates a column as the multiplication of all the multiply_column_names."""
    raise NotImplementedError()

  def drop(self, *drop_columns: str) -> "PandasDataset":
    """Drops the columns named in drop_columns from the dataset."""
    raise NotImplementedError()

  def labels_are_constant(self) -> bool | None:
    """Are the labels constant?

    Returns true if the labels are the same for every row in the dataset,
    false otherwise, and None if there are no labels.
    """
    raise NotImplementedError()

  def label_average(self) -> float | None:
    """Return the average label, weighted by the weights if they exist."""
    raise NotImplementedError()

  def __len__(self) -> int:
    """Return the number of rows in the dataset."""
    raise NotImplementedError()

  def shuffle_inplace(self) -> None:
    """Shuffles the rows of the dataset inplace."""
    raise NotImplementedError()

  def select_features_labels_and_weights(
      self,
      feature_columns: list[str],
      label_column: str | None = None,
      weight_column: str | None = None,
  ) -> "PandasDataset":
    """Returns a dataset with the given features, labels, and weights.

    The features retain their existing column names, but the labels and weights
    are renamed to labels_ and weights_. If the labels or weights are None then
    they are not added.

    Args:
      feature_columns: The feature columns to include in the dataset.
      label_column: The label column to include in the dataset. This will be
        renamed to "label_". If None, then no labels are added.
      weight_column: The weight column to include in the dataset. This will be
        renamed to "weight_". If None, then no weights are added.
    """
    raise NotImplementedError()

  def get_columns(self) -> list[str]:
    """Returns the column names in the dataset."""
    return self.data.columns.values.tolist()
