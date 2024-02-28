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

"""This module contains the base model class.

The base model class is the class that all other base models must inherit from.
It ensures that the models have the methods that are expected by the main
FractionalUpliftModel class.
"""

import abc
from typing import Any, Mapping

import numpy as np

from fractional_uplift import object_duplicator
from fractional_uplift.datasets import _base


Dataset = _base.Dataset


class BaseModel(object_duplicator.GetParamsMixin, abc.ABC):
  """Base model class.

  This is a base class that all base models should inherit from. It defines a
  common API so all models can be trained and make predictions in the same way.

  This ensures that the models all have the methods expected by the main
  FractionalUpliftModel class.
  """
  _label_average: float | None

  def __init__(self):
    """Initialises the model.

    Sets default values for fitted and label average.
    """
    self._label_average = None
    self.fitted = False

  def fit(self, train_data: Dataset, **fit_kw: Any) -> None:
    """Fits the model using the train_data.

    This first checks if the labels are constant. If they are, it doesn't train
    the model, and just uses the average. If the labels are not constant, it
    calls the _fit() method which fits the model.

    Args:
      train_data: The training data for the mode
      **fit_kw: All keyword args are passed to the model fit function
    """
    if train_data.labels_are_constant():
      self._label_average = train_data.label_average()
    else:
      self._fit(train_data, **fit_kw)
    self.fitted = True

  def predict(self, data: Dataset) -> np.ndarray:
    """Makes predictions using the model.

    If the labels were constant in the training data, return that constant.
    Otherwise call the models predict function using _predict().

    Args:
      data: The data to make predictions from.

    Returns:
      The predictions from the model

    Raises:
      RuntimeError: If the model has not been trained yet with .fit().
    """
    if not self.fitted:
      raise RuntimeError("Model has not been trained, cannot predict.")

    if self._label_average is not None:
      return self._label_average * np.ones(len(data), dtype=np.float32)
    else:
      preds = self._predict(data)
      shape = np.shape(preds)
      if len(shape) == 1:
        return preds
      elif shape[1] > 1:
        return preds
      else:
        return preds.flatten()

  @abc.abstractmethod
  def _fit(self, train_data: Dataset, **fit_kw: Any) -> None:
    """Fits the model using the train_data.

    This should be entirely automated. If the model should do any
    hyperparameter tuning or early stopping on validation data, then that must
    be implemented here, and the validation data must be automatically split
    from the train_data inside this method.

    Args:
      train_data: The training data for the model
      **fit_kw: All keyword args are passed to the model fit function
    """
    ...

  @abc.abstractmethod
  def _predict(self, data: Dataset) -> np.ndarray:
    """Makes predictions using the model.

    Args:
      data: The data to make predictions from.
    """
    ...

  @abc.abstractmethod
  def export(self) -> Any:
    """Exports the underlying model.

    Returns:
      The underlying model for integration with external
        pipelines outside of this package.
    """
    ...
