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

"""A module containing the base classes for the datasets.

Base classes ensuring that all the different datasets used are standardised.
"""

import abc
from typing import Any, Optional

import pandas as pd
import tensorflow as tf

from fractional_uplift import constants
from fractional_uplift import object_duplicator


KPI = constants.KPI


class Dataset(object_duplicator.GetParamsMixin, abc.ABC):
  """A base class for a dataset which can be accessed in multiple ways.

  This can be accessed as either a pandas dataframe or a tensorflow dataset.
  """

  def __init__(self, **params: Any):
    self._store_params(**params)

  @abc.abstractmethod
  def as_pd_dataframe(self) -> pd.DataFrame:
    """Returns the dataset as a pandas dataframe.

    If they exist, the labels are in the column named "label_" and weights in a
    column named "weight_".
    """
    ...

  @abc.abstractmethod
  def as_tf_dataset(self) -> tf.data.Dataset:
    """Returns the dataset as a tensorflow dataset object."""
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
    """Return the average label."""
    ...

  @abc.abstractmethod
  def __len__(self) -> int:
    ...


class TrainData(abc.ABC):
  """An instance of training data for uplift models.

  Contains all the required datasets depending on the problem.
  """

  def __init__(self):
    self._data_cache = dict()

  def get_inc_classifier_data(self, kpi: KPI) -> Optional[Dataset]:
    """Returns the dataset to train the incrementality classifier model.

    If the kpi does not exist, it returns None.

    Automatically caches the result so it doesn't create multiple copies if
    called multiple times.

    Args:
      kpi: The kpi to get the dataset for.
    """
    key = f"get_inc_classifier_data__{kpi.name}"
    if key not in self._data_cache:
      self._data_cache[key] = self._get_inc_classifier_data(kpi)

    return self._data_cache[key]

  def get_data_for_single_treatment(
      self, kpi: KPI, is_treated: bool
  ) -> Optional[Dataset]:
    """Returns the dataset for specified value of is_treated.

    If the kpi does not exist, it returns None.

    Automatically caches the result so it doesn't create multiple copies if
    called multiple times.

    Args:
      kpi: The kpi to get the dataset for.
      is_treated: Get the treatment group (True) or control group (False).
    """

    key = f"get_treatment_data__{kpi.name}__{is_treated}"
    if key not in self._data_cache:
      self._data_cache[key] = self._get_data_for_single_treatment(
          kpi, is_treated
      )

    return self._data_cache[key]

  def get_kpi_weight_data(self) -> Optional[Dataset]:
    """Returns the dataset to train the kpi weight model.

    If the only kpi that is set is the maximize_kpi kpi, then this
    will return None, as no kpi weight model is needed.

    Automatically caches the result so it doesn't create multiple copies if
    called multiple times.
    """
    if "get_kpi_weight_data" not in self._data_cache:
      self._data_cache["get_kpi_weight_data"] = self._get_kpi_weight_data()

    return self._data_cache["get_kpi_weight_data"]

  @abc.abstractmethod
  def _get_inc_classifier_data(self, kpi: KPI) -> Optional[Dataset]:
    """Returns the dataset to train an incrementality classifier model.

    If either the kpi does not exist this should return None.

    Must be defined in the subclass. Do not use this directly, instead use
    get_inc_classifier_data() which caches the result automatically.

    Args:
      kpi: The kpi to get the dataset for.
    """
    ...

  @abc.abstractmethod
  def _get_kpi_weight_data(self) -> Optional[Dataset]:
    """Returns the dataset to train the kpi weight model.

    If the only kpi that is set is the maximize_kpi kpi, then this
    should return None, as no kpi weight model is needed.

    Must be defined in the subclass. Do not use this directly, instead use
    get_kpi_weight_data() which caches the result automatically.
    """
    ...

  @abc.abstractmethod
  def _get_data_for_single_treatment(
      self, kpi: KPI, is_treated: bool
  ) -> Optional[Dataset]:
    """Returns the dataset for specfied value of is_treated.

    If the kpi does not exist, it returns None.

    Args:
      kpi: The kpi to get the dataset for.
      is_treated: Get the treatment group (True) or control group (False).
    """
    ...

  @abc.abstractmethod
  def has_kpi(self, kpi: KPI) -> bool:
    """Returns True if the kpi exists in this dataset."""
    ...

  @abc.abstractmethod
  def has_non_negative_kpi(self, kpi: KPI) -> bool:
    """Returns True if the kpi exists in this dataset and is non-negative."""
    ...
