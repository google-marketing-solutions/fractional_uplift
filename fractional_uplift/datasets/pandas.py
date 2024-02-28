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

"""A module containing the datasets which load from pandas.

This module contains the classes needed to construct uplift modelling datasets
from a pandas dataframe.
"""

import dataclasses
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_decision_forests as tfdf

from fractional_uplift import constants
from fractional_uplift.datasets import _base


ColumnName = constants.ColumnName
KPI = constants.KPI
MAX_N_CLASSES = constants.MAX_N_CLASSES
Task = constants.Task
TaskType = constants.TaskType

Dataset = _base.Dataset
TrainData = _base.TrainData


def cast_pandas_dataframe_cols(
    input_data: pd.DataFrame,
    cat_feature_cols: Sequence[str],
    num_feature_cols: Sequence[str],
) -> pd.DataFrame:
  """Casts the features to the appropriate types.

  If the feature is in cat_feature_cols then cast it to
  a "category" datatype. If it's in num_feature_cols then cast it to
  a float64 data type. Otherwise ignore it.

  Args:
    input_data: The pandas dataframe to cast the columns of.
    cat_feature_cols: The list of feature names to be treated as categorical
    num_feature_cols: The list of feature names to be treated as numerical

  Returns:
    A dataframe with the columns casted to categorical or float.

  Raises:
    ValueError: If cat_feature_cols and num_feature_cols overlap
  """
  overlap_feature_cols = set(cat_feature_cols) & set(num_feature_cols)
  if overlap_feature_cols:
    raise ValueError(
        "The following columns are se in both cat_feature_cols and"
        f" num_feature_cols: {overlap_feature_cols}"
    )

  skip_cols = (
      set(input_data.columns.values)
      - set(num_feature_cols)
      - set(cat_feature_cols)
  )
  untransformed_data = input_data[list(skip_cols)].copy()
  num_data = input_data[num_feature_cols].astype(np.float64)
  cat_data = input_data[cat_feature_cols].astype("category")

  return pd.concat([untransformed_data, num_data, cat_data], axis=1)[
      input_data.columns
  ]


class PandasDataset(Dataset):
  """A single dataset which has been created from a pandas dataframe.

  The data is stored internally as a pandas dataframe, and is converted into
  the required format when the corresponding method is called. Current methods
  supported are:
    - as_pd_dataframe()
    - as_tf_dataset()
  """

  def __init__(
      self,
      features_data: pd.DataFrame,
      *,
      labels: Optional[np.ndarray] = None,
      weights: Optional[np.ndarray] = None,
      shuffle: bool = False,
      shuffle_seed: Optional[int] = None,
      copy: bool = True,
      task: Optional[TaskType] = None,
  ):
    """Constructs a dataset from a pandas dataframe.

    This will name the label and weight columns properly. It expects that all
    categorical features are categorical type, and all numerical features are
    float64 type. You can set this with the helper function
    cast_pandas_dataframe_cols().

    Args:
      features_data: A dataframe containing all the features.
      labels: The labels for the classification problem (max 3 classes)
      weights: The sample_weights of the classification problem
      shuffle: If true, shuffle the dataset
      shuffle_seed: Seed used for shuffling, to make it deterministic.
      copy: If the features_data, labels and weights should be copied
      task: Either Task.CLASSIFICATION, Task.REGRESSION or None if there is no
        label

      Raises:
        ValueError: If the labels are set without a task, or if the weights are
        set without a label
    """
    super().__init__(
        features_data=features_data,
        labels=labels,
        weights=weights,
        shuffle=shuffle,
        shuffle_seed=shuffle_seed,
        copy=copy,
        task=task,
    )
    self._validate_feature_cols(features_data)

    if copy:
      features_data = features_data.copy()

    self._data = features_data.reset_index(drop=True)

    accepted_tasks = {Task.CLASSIFICATION, Task.REGRESSION, None}
    if task not in accepted_tasks:
      raise ValueError(f"task must be one of {accepted_tasks}")
    self._task = task

    if labels is not None:
      if task is None:
        raise ValueError("Must set task if also setting labels")
      if copy:
        labels = labels.copy()
      self._add_labels_to_data(labels)

    if weights is not None:
      if labels is None:
        raise ValueError("Must set labels if also setting weights")
      if copy:
        weights = weights.copy()
      self._add_weights_to_data(weights)

    if shuffle:
      self._data = self._data.sample(frac=1.0, random_state=shuffle_seed)

  def _add_labels_to_data(self, labels: np.ndarray) -> None:
    """Adds the labels to the dataframe.

    The labels must be categorical labels, with max 3 classes. This is all that
    is required for the fractional uplift model.

    Args:
      labels: Numpy array containing the labels
    """
    if np.shape(labels) != np.shape(self._data.index.values):
      raise ValueError(
          f"Wrong shape of labels. Got {np.shape(labels)}, "
          f"expected {np.shape(self._data.index.values)}."
      )

    if self._task == Task.CLASSIFICATION:
      if len(np.unique(labels)) > MAX_N_CLASSES:
        raise ValueError(
            f"Labels must have no more than {MAX_N_CLASSES} categories"
        )
      self._data[ColumnName.LABEL.value] = pd.Categorical(labels)
    elif self._task == Task.REGRESSION:
      self._data[ColumnName.LABEL.value] = pd.Series(labels).astype(np.float64)
    else:
      raise AssertionError(f"Unexpected task: {self._task}")

  def _add_weights_to_data(self, weights: np.ndarray) -> None:
    """Adds the weights to the dataframe.

    These weights will be used as sample_weights in the classification problem.
    They will be set by the PandasTrainData class.

    If the weights are zero, those rows will be filtered out.

    Args:
      weights: Numpy array containing the weights
    """
    if np.shape(weights) != np.shape(self._data.index.values):
      raise ValueError(
          f"Wrong shape of weights. Got {np.shape(weights)}, "
          f"expected {np.shape(self._data.index.values)}."
      )
    if np.any(weights < 0.0):
      raise ValueError("Weight must be >= 0")
    if np.any(~np.isfinite(weights)):
      raise ValueError("Weight must be finite")

    self._data[ColumnName.WEIGHT.value] = weights
    self._data = self._data.loc[self._data[ColumnName.WEIGHT.value] > 0.0]

  def _validate_feature_cols(self, data: pd.DataFrame) -> None:
    """Checks that the cols in the input pandas dataframe are valid.

    Checks that the names do not overlap with the protected names for the
    weights and labels columns, and that they are all either float64 or
    categorical type.

    Args:
      data: The pandas dataframe to validate

    Raises:
      ValueError: If data has columns named 'label_' or 'weight_'
    """
    if ColumnName.WEIGHT.value in data.columns:
      raise ValueError(
          f"Must not have the protected name {ColumnName.WEIGHT.value} "
          "as a feature name."
      )
    if ColumnName.LABEL.value in data.columns:
      raise ValueError(
          f"Must not have the protected name {ColumnName.LABEL.value} "
          "as a feature name."
      )

    bad_columns = [
        column
        for column in data.columns
        if data[column].dtype not in ["category", "float64"]
    ]

    if bad_columns:
      raise ValueError(
          f"Columns {bad_columns} are not categorical or float64 dtypes."
      )

  def as_pd_dataframe(self) -> pd.DataFrame:
    """Returns the dataset as a pandas dataframe.

    If they exist, the labels are in the column named "label_" and weights in a
    column named "weight_".
    """
    return self._data

  def as_tf_dataset(self) -> tf.data.Dataset:
    """Returns the dataset as a tensorflow dataset object."""
    args = {"task": self._task}
    if ColumnName.LABEL.value in self._data.columns:
      args["label"] = ColumnName.LABEL.value
    if ColumnName.WEIGHT.value in self._data.columns:
      args["weight"] = ColumnName.WEIGHT.value

    return tfdf.keras.pd_dataframe_to_tf_dataset(self._data, **args)

  def labels_are_constant(self) -> bool | None:
    """Are the labels constant?

    Returns:
      True if the labels are the same for every row in the dataset,
        false otherwise, and None if there are no labels.
    """
    if ColumnName.LABEL.value not in self._data.columns:
      return None

    if len(self) < 2:
      return True

    if self._task == Task.CLASSIFICATION:
      n_categories = len(np.unique(self._data[ColumnName.LABEL.value]))
      return n_categories == 1

    if self._task == Task.REGRESSION:
      standard_deviation = np.std(self._data[ColumnName.LABEL.value].values)
      return np.isclose(standard_deviation, 0.0, atol=1e-7)

  def label_average(self) -> float | None:
    """Return the average label.

    This is the mean of the label if it's a regression task, and the mode if
    it's a classification task.
    """
    if ColumnName.LABEL.value not in self._data.columns:
      return None

    if self._task == Task.CLASSIFICATION:
      return float(self._data[ColumnName.LABEL.value].mode())

    if self._task == Task.REGRESSION:
      return self._data[ColumnName.LABEL.value].mean()

  def __len__(self) -> int:
    return len(self._data.index.values)


@dataclasses.dataclass
class PandasTrainData(TrainData):
  """An instance of training data for uplift models, from a pandas dataframe.

  Contains all the required datasets depending on the uplift modelling problem.
  See the docstring on the FractionalUpliftModel for more information.
  """
  features_data: pd.DataFrame
  maximize_kpi: np.ndarray
  is_treated: np.ndarray
  treatment_propensity: np.ndarray

  constraint_kpi: Optional[np.ndarray] = None
  constraint_offset_kpi: Optional[np.ndarray] = None
  sample_weight: Optional[np.ndarray] = None
  shuffle_seed: Optional[int] = None

  def __post_init__(self) -> None:
    """Validates the parameters have been set correctly."""
    super().__init__()
    self.kpis = {
        KPI.MAXIMIZE_KPI.name: self.maximize_kpi,
        KPI.CONSTRAINT_KPI.name: self.constraint_kpi,
        KPI.CONSTRAINT_OFFSET_KPI.name: self.constraint_offset_kpi,
    }

    if (self.constraint_offset_kpi is not None) & (self.constraint_kpi is None):
      raise ValueError(
          "Must set constraint_kpi if also setting constraint_offset_kpi."
      )

    if self.sample_weight is None:
      self.sample_weight = np.ones_like(self.maximize_kpi)

    if np.any(self.sample_weight < 0.0):
      raise ValueError("Sample weights must be >= 0")

    if not np.all((self.is_treated == 0) | (self.is_treated == 1)):
      raise ValueError("is_treated must be 0 or 1")

    if not np.all(
        (self.treatment_propensity > 0.0) & (self.treatment_propensity < 1.0)
    ):
      raise ValueError("Treatment_propensity must always be > 0 and < 1")

  @property
  def inverse_propensity_weights(self) -> np.ndarray:
    """Returns the inverse propensity weights.

    This is based on is_treated and treatment_propensity. The sample is weighed
    inversely to the propensity that it would have been assiged to its
    treatment group.

    If is_treated = 1, it's weighed by 1 / treatment_propensity
    If is_treated = 0, it's weighed by 1 / (1 - treatment_propensity)
    """
    ipw_control = self.is_treated  / self.treatment_propensity
    ipw_treated = (1 - self.is_treated) / (1.0 - self.treatment_propensity)
    ipw = ipw_control + ipw_treated
    return ipw

  def _get_inc_classifier_data(self, kpi: KPI) -> Optional[Dataset]:
    """Returns the dataset to train an incrementality classifier model.

    If the kpi does not exist this should return None.

    Must be defined in the subclass. Do not use this directly, instead use
    get_inc_classifier_data() which caches the result automatically.

    Args:
      kpi: The kpi to get the dataset for.
    """

    kpi_values = self.kpis[kpi.name]

    if kpi_values is None:
      # KPI does not exist
      return None

    weights = kpi_values * self.inverse_propensity_weights * self.sample_weight
    return PandasDataset(
        self.features_data,
        labels=self.is_treated,
        weights=weights,
        shuffle=True,
        shuffle_seed=self.shuffle_seed,
        task=Task.CLASSIFICATION,
    )

  def _get_data_for_single_treatment(
      self, kpi: KPI, is_treated: bool
  ) -> Optional[Dataset]:
    """Returns the dataset for specfied value of is_treated.

    If the kpi does not exist, it returns None.

    Args:
      kpi: The kpi to get the dataset for.
      is_treated: Get the treatment group (True) or control group (False).
    """
    kpi_values = self.kpis[kpi.name]

    if kpi_values is None:
      # KPI does not exist
      return None

    # is_treated is an int in the data, not bool
    treatment_mask = self.is_treated == int(is_treated)

    weights = (
        self.inverse_propensity_weights[treatment_mask]
        * self.sample_weight[treatment_mask]
    )

    return PandasDataset(
        self.features_data.loc[treatment_mask],
        labels=kpi_values[treatment_mask],
        weights=weights,
        shuffle=True,
        shuffle_seed=self.shuffle_seed,
        task=Task.REGRESSION,
    )

  def _get_kpi_weight_data(self) -> Optional[Dataset]:
    """Returns the dataset to train the kpi weight model.

    If the only kpi that is set is the maximize_kpi kpi, then this
    should return None, as no kpi weight model is needed.
    """
    if not self.has_kpi(KPI.CONSTRAINT_KPI):
      # We only have the maximize_kpi kpi, so no model needed
      return None

    data_lst = [self.features_data]
    labels_lst = [KPI.MAXIMIZE_KPI.value] * len(self.maximize_kpi)
    weights_lst = [
        self.maximize_kpi * self.inverse_propensity_weights * self.sample_weight
    ]

    data_lst.append(self.features_data)
    labels_lst += [KPI.CONSTRAINT_KPI.value] * len(self.constraint_kpi)
    weights_lst.append(
        self.constraint_kpi
        * self.inverse_propensity_weights
        * self.sample_weight
    )

    if self.constraint_offset_kpi is not None:
      data_lst.append(self.features_data)
      labels_lst += [KPI.CONSTRAINT_OFFSET_KPI.value] * len(
          self.constraint_offset_kpi
      )
      weights_lst.append(
          self.constraint_offset_kpi
          * self.inverse_propensity_weights
          * self.sample_weight
      )

    features_data = pd.concat(data_lst).reset_index(drop=True)
    labels = np.array(labels_lst)
    weights = np.concatenate(weights_lst)

    return PandasDataset(
        features_data,
        labels=labels,
        weights=weights,
        shuffle=True,
        shuffle_seed=self.shuffle_seed,
        task=Task.CLASSIFICATION,
    )

  def has_kpi(self, kpi: KPI) -> bool:
    """Returns True if the kpi exists in this dataset."""
    return self.kpis.get(kpi.name) is not None

  def has_non_negative_kpi(self, kpi: KPI) -> bool:
    """Returns True if the kpi exists in this dataset and is non-negative."""
    if self.kpis.get(kpi.name) is None:
      return False

    return np.all(self.kpis.get(kpi.name) >= 0.0)
