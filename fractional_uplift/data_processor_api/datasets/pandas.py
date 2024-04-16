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

from fractional_uplift import constants
from fractional_uplift.data_processor_api.datasets import base


ColumnName = constants.ColumnName


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
    """Creates a column as the addition of the addition_column_names."""
    if not addition_column_names:
      raise ValueError("The addition column names cannot be empty.")

    self.data[output_column_name] = 0.0
    for addition_column_name in addition_column_names:
      self.data[output_column_name] += self.data[addition_column_name]

    return self

  def set_column_from_division(
      self,
      output_column_name: str,
      numerator_column_name: str,
      denominator_column_name: str,
  ) -> "PandasDataset":
    """Creates a column as the division of the numerator and denominator."""
    self.data[output_column_name] = (
        self.data[numerator_column_name] / self.data[denominator_column_name]
    )
    return self

  def set_column_from_multiplication(
      self, output_column_name: str, *multiply_column_names: str
  ) -> "PandasDataset":
    """Creates a column as the multiplication of all the multiply_column_names."""
    if not multiply_column_names:
      raise ValueError("The multiply column names cannot be empty.")

    self.data[output_column_name] = 1.0
    for multiply_column_name in multiply_column_names:
      self.data[output_column_name] *= self.data[multiply_column_name]

    return self

  def drop(self, *drop_columns: str) -> "PandasDataset":
    """Drops the columns named in drop_columns from the dataset."""
    self.data.drop(columns=list(drop_columns), inplace=True)
    return self

  def select_features_labels_and_weights(
      self,
      feature_columns: list[str],
      *,
      label_column: str | None = None,
      weight_column: str | None = None,
  ) -> "PandasDataset":
    """Returns a dataset with the given features, labels, and weights.

    The features retain their existing column names, but the labels and weights
    are renamed to labels_ and weights_. If the labels or weights are None then
    they are not added.

    Args:
      feature_columns: The feature columns to include in the dataset. This
        cannot be empty.
      label_column: The label column to include in the dataset. This will be
        renamed to "label_". If None, then no labels are added. The label column
        must always be numeric. For classification problems, the label should be
        an integer with a different value for each class, while for regression
        it can be either integer or float.
      weight_column: The weight column to include in the dataset. This will be
        renamed to "weight_". If None, then no weights are added. The weight
        column must be numeric and non-negaitve.

    Raises:
      ValueError: if either "label_" or "weight_" is one of the features.
      ValueError: if the weights are non-numeric or negative.
      ValueError: if the label is non-numeric.
      ValueError: if the feature columns lits is empty.
      ValueError: if there is any overlap between the label, weight and feature
        columns.
    """
    if ColumnName.LABEL.value in feature_columns:
      raise ValueError("The features cannot contain a column named 'label_'")
    if ColumnName.WEIGHT.value in feature_columns:
      raise ValueError("The features cannot contain a column named 'weight_'")

    if not feature_columns:
      raise ValueError("The feature columns cannot be empty")

    non_none_labels_and_weights = {}
    if label_column:
      non_none_labels_and_weights[label_column] = ColumnName.LABEL.value
      if not self.column_is_numeric(label_column):
        raise ValueError("The label column must be numeric.")
    if weight_column:
      non_none_labels_and_weights[weight_column] = ColumnName.WEIGHT.value
      if not self.column_is_numeric(weight_column):
        raise ValueError("The weight column must be numeric.")
      if not self.column_is_not_negative(weight_column):
        raise ValueError("The weight column must be non-negative.")

    select_columns = feature_columns + list(non_none_labels_and_weights.keys())

    unique_columns = set(select_columns)
    if len(unique_columns) < len(select_columns):
      raise ValueError(
          "There are duplicate columns. Check you are not using the same column"
          " name more than once across feaures, labels or weights."
      )

    self.data = self.data[select_columns]
    self.data = self.data.rename(columns=non_none_labels_and_weights)

    return self

  def labels_are_constant(self) -> bool | None:
    """Returns true if the labels are constant, or None if they do not exist.

    The labels are found in the column named "label_", which is set with
    select_features_labels_and_weights().
    """
    if not self.column_exists(ColumnName.LABEL.value):
      return None

    label_values = self.data[ColumnName.LABEL.value].values
    return np.allclose(label_values, label_values[0])

  def label_average(self) -> float | None:
    """Return the average label, weighted by the weights if they exist.

    The labels are found in the column named "label_", and the weights are
    found in the column named "weight_", which are set with
    select_features_labels_and_weights().

    If the column label_ does not exist, then this returs None.
    """
    if self.column_exists(ColumnName.LABEL.value):
      labels = self.data[ColumnName.LABEL.value].values
    else:
      return None

    if self.column_exists(ColumnName.WEIGHT.value):
      weights = self.data[ColumnName.WEIGHT.value].values
    else:
      weights = np.ones(len(self.data))

    return np.sum(labels * weights) / np.sum(weights)

  def __len__(self) -> int:
    """Return the number of rows in the dataset."""
    return len(self.data)

  def shuffle_inplace(self, random_seed: int = 0) -> None:
    """Shuffles the rows of the dataset inplace."""
    self.data = self.data.sample(
        frac=1.0, ignore_index=True, random_state=random_seed
    )

  def get_columns(self) -> list[str]:
    """Returns the column names in the dataset."""
    return self.data.columns.values.tolist()

  def column_is_finite(self, column_name: str) -> bool:
    """Returns true if the column is finite for all rows."""
    return np.all(np.isfinite(self.data[column_name]))

  def column_is_numeric(self, column_name: str) -> bool:
    """Returns true if the column is float or int for all rows."""
    return pd.api.types.is_numeric_dtype(self.data[column_name])
