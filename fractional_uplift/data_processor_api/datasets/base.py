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

import abc
from typing import Any

import numpy as np
import pandas as pd


class Dataset(abc.ABC):
  """A base class for a dataset.

  The dataset can perform a variety of data processing actions, and can export
  the dataset to a varity of formats. It must be subclassed, and each subclass
  will internally store and process the data using different packages, such as
  pandas.

  Subclassing this allows you to integrate Fractional Uplift with any data
  processing tech stack.

  In addition to defining all the abstract methods, you must define an __init__
  method, which takes as an input the actual data for this dataset. Then all
  the methods must be defined to make changes to this data using your data
  processing package.
  """

  @abc.abstractmethod
  def column_is_not_negative(self, column_name: str) -> bool:
    """Returns true if the column is not negative for all rows."""
    ...

  @abc.abstractmethod
  def column_exists(self, column_name: str) -> bool:
    """Returns true if the column exists in the dataset."""
    ...

  @abc.abstractmethod
  def copy(self) -> "Dataset":
    """Returns a copy of the dataset."""
    ...

  @abc.abstractmethod
  def set_column_from_constant(
      self, column_name: str, column_value: Any
  ) -> "Dataset":
    """Creates a column with the given name and constant value."""
    ...

  @abc.abstractmethod
  def set_column_from_numpy_array(
      self, column_name: str, column_values: np.ndarray
  ) -> "Dataset":
    """Creates a column from a numpy array."""
    ...

  @abc.abstractmethod
  def filter(self, mask_column: str) -> "Dataset":
    """Filters the dataset to only rows where the mask_column is true."""
    ...

  @abc.abstractmethod
  def append_rows(self, data: "Dataset") -> "Dataset":
    """Appends the rows of the given dataset to the end of this dataset.

    This ignores the index of the pandas dataframe, so the index gets reset.

    Args:
      data: The dataset to append to this dataset.

    Returns:
      The concatenated dataset.

    Raises:
      ValueError: If the columns don't match between the two datasets.
    """
    ...

  @abc.abstractmethod
  def set_column_from_subtraction(
      self,
      *,
      output_column_name: str,
      minuend_column_name: str,
      subtrahend_column_name: str,
  ) -> "Dataset":
    """Creates a column as the subtraction of the subtrahend from the minuend."""
    ...

  @abc.abstractmethod
  def set_column_from_addition(
      self, output_column_name: str, *addition_column_names: str
  ) -> "Dataset":
    """Creates a column that is the sum of the addition column names."""
    ...

  @abc.abstractmethod
  def set_column_from_division(
      self,
      output_column_name: str,
      numerator_column_name: str,
      denominator_column_name: str,
  ) -> "Dataset":
    """Creates a column as the division of the numerator and denominator."""
    ...

  @abc.abstractmethod
  def set_column_from_multiplication(
      self, output_column_name: str, *multiply_column_names: str
  ) -> "Dataset":
    """Creates a column as the multiplication of all the multiply_column_names."""
    ...

  @abc.abstractmethod
  def drop(self, *drop_columns: str) -> "Dataset":
    """Drops the columns named in drop_columns from the dataset."""
    ...

  @abc.abstractmethod
  def labels_are_constant(self) -> bool | None:
    """Are the labels constant?

    Returns true if the labels are the same for every row in the dataset,
    false otherwise, and None if there are no labels.
    """
    ...

  @abc.abstractmethod
  def label_average(self) -> float | None:
    """Return the average label, weighted by the weights if they exist."""
    ...

  @abc.abstractmethod
  def __len__(self) -> int:
    """Return the number of rows in the dataset."""
    ...

  @abc.abstractmethod
  def shuffle_inplace(self) -> None:
    """Shuffles the rows of the dataset inplace."""
    ...

  @abc.abstractmethod
  def as_pd_dataframe(self) -> pd.DataFrame:
    """Returns the dataset as a pandas dataframe."""
    ...

  @abc.abstractmethod
  def select_features_labels_and_weights(
      self,
      feature_columns: list[str],
      label_column: str | None = None,
      weight_column: str | None = None,
  ) -> "Dataset":
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
    ...

  @abc.abstractmethod
  def get_columns(self) -> list[str]:
    """Returns the column names in the dataset."""
    ...
