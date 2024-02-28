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

"""A module containing the base tensorflow decision forest base models.

This module contains a class to create an uplift model from a tensorflow
decision forest model.
"""

import abc
from typing import Any, Dict

import numpy as np
import tensorflow_decision_forests as tfdf

from fractional_uplift.base_models import _base as base_models_base
from fractional_uplift.datasets import _base as datasets_base


Dataset = datasets_base.Dataset
BaseModel = base_models_base.BaseModel
Task = tfdf.keras.core.Task
TaskType = tfdf.keras.core.TaskType


class TensorflowDecisionForestBase(BaseModel, metaclass=abc.ABCMeta):
  """Abstract base class for tensorflow decision forests.

  Doesn't add much functionality to the decision forest used, except changes the
  signature of the methods to accept a Dataset object.

  Do not use this directly, use TensorflowDecisionForestClassifier or
  TensorflowDecisionForestRegressor instead.
  """
  DEFAULT_INIT_ARGS = {"verbose": 0}
  DEFAULT_FIT_ARGS = {"verbose": 0}

  def __init__(
      self,
      decision_forest_cls: type[tfdf.keras.CoreModel] | None = None,
      init_args: Dict[str, Any] | None = None,
      fit_args: Dict[str, Any] | None = None,
  ) -> None:
    """Initializes the model.

    Must not specify "task" in init_args.

    Args:
      decision_forest_cls: The base decision forest model class to be used, from
        tensorflow decision forests. If set to None, then a
        GradientBoostedTreesModel is used.
      init_args: The args used to instantiate the decision forest
      fit_args: The args used to fit the decision forest
    """
    super().__init__()

    self._store_params(
        decision_forest_cls=decision_forest_cls,
        init_args=init_args,
        fit_args=fit_args,
    )
    decision_forest_cls = (
        decision_forest_cls or tfdf.keras.GradientBoostedTreesModel
    )

    fit_args = self.DEFAULT_FIT_ARGS.copy() | (fit_args or {})
    init_args = self.DEFAULT_INIT_ARGS.copy() | (init_args or {})

    if "task" in init_args:
      raise ValueError(
          f"Cannot set task with init_args. It is always set to {self._task}."
      )
    init_args["task"] = self._task

    self._decision_forest = decision_forest_cls(**init_args)
    self._fit_args = fit_args

  @property
  @abc.abstractmethod
  def _task(self) -> TaskType:
    ...

  def _fit(self, train_data: Dataset) -> None:
    """Fits the model using the train_data."""
    self._decision_forest.fit(train_data.as_tf_dataset(), **self._fit_args)

  def _predict(self, data: Dataset) -> np.ndarray:
    """Makes predictions using the model."""
    return self._decision_forest.predict(data.as_tf_dataset())

  def export(self) -> tfdf.keras.CoreModel:
    """Exports the underlying tensorflow decision forest model."""
    return self._decision_forest


class TensorflowDecisionForestClassifier(TensorflowDecisionForestBase):
  """A class to train and predict using a tensorflow decision forest classifier.

  Doesn't add much functionality to the decision forest used, except changes the
  signature of the methods to accept a Dataset object.
  """

  _task = Task.CLASSIFICATION


class TensorflowDecisionForestRegressor(TensorflowDecisionForestBase):
  """A class to train and predict using a tensorflow decision forest classifier.

  Doesn't add much functionality to the decision forest used, except changes the
  signature of the methods to accept a Dataset object.
  """

  _task = Task.REGRESSION
