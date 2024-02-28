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

"""The main classes to implement uplift modelling.

This module contains the classes to implement uplift modelling. They
can use any base model, and any dataset.

Current it contains the following uplift models:
  - RetrospectiveEstimationUpliftModel(). This is for learning the relative
  uplift of one KPI.
  - FractionalUpliftModel(): This is for using uplift modelling to solve the
      fractional knapsack problem. Optimal for maximising a KPI with an
      RoI constraint (amongst other use-cases). Discussed in detail in the
      readme.
"""

import abc
from typing import Any, Optional, Type

import numpy as np

from fractional_uplift import constants
from fractional_uplift import object_duplicator
from fractional_uplift.base_models import _base as base_models_base
from fractional_uplift.datasets import _base as datasets_base


Dataset = datasets_base.Dataset
TrainData = datasets_base.TrainData
BaseModel = base_models_base.BaseModel
KPI = constants.KPI
Task = constants.Task


class BaseSingleKPILearner(abc.ABC):
  """A base class for all single KPI meta learners."""

  def __init__(
      self,
      base_model: BaseModel,
      target_kpi: KPI = KPI.MAXIMIZE_KPI,
  ):
    """Initialises the learner.

    Args:
      base_model: The model to predict the incrementality of the target kpi.
      target_kpi: Either 'KPI.MAXIMIZE_KPI', 'KPI.CONSTRAINT_KPI' or
        'KPI.CONSTRAINT_OFFSET_KPI'.
    """
    self.fitted = False
    self._setup_base_models(base_model)
    self.target_kpi = target_kpi

  @abc.abstractmethod
  def _setup_base_models(self, base_model: BaseModel):
    """Initialises the base models."""
    ...

  def fit(self, data: TrainData) -> None:
    """Fits the meta learner to the data.

    Args:
      data: The data to train the models on.

    Raises:
      RuntimeError: If the learner has already been trained with .fit()
    """
    if self.fitted:
      raise RuntimeError("Learner is already trained.")

    self._fit(data)
    self.fitted = True

  def predict(self, data: Dataset) -> np.ndarray:
    """Predicts the uplift score.

    Args:
      data: The data to apply the model to.

    Returns:
      Uplift score predictions.

    Raises:
      RuntimeError: If the learner has not been trained yet with .fit()
    """
    if not self.fitted:
      raise RuntimeError("Learner is not trained.")

    return self._predict(data)

  @abc.abstractmethod
  def _fit(self, data: TrainData) -> None:
    """Must subclass and implement fitting the learner.

    Args:
      data: The data to train the models on.
    """
    ...

  @abc.abstractmethod
  def _predict(self, data: Dataset) -> np.ndarray:
    """Must subclass and implement predicting with the learner.

    Args:
      data: The data to apply the model to.

    Returns:
      Uplift score predictions.
    """
    ...

  def distill(
      self, distill_data: Dataset, distill_model: BaseModel, **fit_kw: Any
  ) -> None:
    """Distills the uplift model into a single model.

    This will call fit on the distill_model, so afterwards that model will
    be trained. The distill_data used to distill the model should ideally not be
    the same as the training data, to reduce overfitting.

    Args:
      distill_data: The data to use to train the distill_model.
      distill_model: The model used for distilling.
      **fit_kw: Keyword args that are passed to the fit method of distill_model.
    """
    preds = self.predict(distill_data)
    distill_data_with_labels = object_duplicator.duplicate_object(
        distill_data,
        labels=preds,
        copy=True,
        task=Task.REGRESSION,
    )
    distill_model.fit(distill_data_with_labels, **fit_kw)


class BaseFractionalLearner(abc.ABC):
  """A base class for all fractional meta learners."""

  def __init__(self):
    """Initialises the meta learner."""
    self.fitted = False

  def fit(self, data: TrainData) -> None:
    """Fits the meta learner to the data.

    Args:
      data: The data to train the models on.

    Raises:
      RuntimeError: If the learner has already been trained with .fit()
    """
    if self.fitted:
      raise RuntimeError("Learner is already trained.")

    self._fit(data)
    self.fitted = True

  def predict(
      self, data: Dataset, constraint_offset_scale: float | None = None
  ) -> np.ndarray:
    """Predicts the uplift score.

    Args:
      data: The data to apply the model to.
      constraint_offset_scale: The scale for the constraint_offset_kpi. Not
        needed if constraint_offset_kpi is not provided.

    Returns:
      Uplift score predictions.

    Raises:
      RuntimeError: If the learner has not been trained yet with .fit()
    """
    if not self.fitted:
      raise RuntimeError("Learner is not trained.")

    return self._predict(data, constraint_offset_scale)

  @abc.abstractmethod
  def _fit(self, data: TrainData) -> None:
    """Must subclass and implement fitting the learner.

    Args:
      data: The data to train the models on.
    """
    ...

  @abc.abstractmethod
  def _predict(
      self, data: Dataset, constraint_offset_scale: float | None = None
  ) -> np.ndarray:
    """Must subclass and implement predicting with the learner.

    Args:
      data: The data to apply the model to.
      constraint_offset_scale: The scale for the constraint_offset_kpi. Not
        needed if constraint_offset_kpi is not provided.

    Returns:
      Uplift score predictions.
    """
    ...

  def distill(
      self,
      distill_data: Dataset,
      distill_model: BaseModel,
      constraint_offset_scale: float | None = None,
      **fit_kw: Any,
  ) -> None:
    """Distills the uplift model into a single model.

    This will call fit on the distill_model, so afterwards that model will
    be trained. The distill_data used to distill the model should ideally not be
    the same as the training data, to reduce overfitting.

    Args:
      distill_data: The data to use to train the distill_model.
      distill_model: The model used for distilling.
      constraint_offset_scale: The constraint_offset_scale to use for
        prediction.
      **fit_kw: Keyword args that are passed to the fit method of distill_model.
    """
    preds = self.predict(
        distill_data, constraint_offset_scale=constraint_offset_scale
    )
    if np.all(preds == np.inf):
      preds[preds == np.inf] = 0.0
    elif np.any(preds == np.inf):
      preds[preds == np.inf] = np.max(preds[np.isfinite(preds)])

    distill_data_with_labels = object_duplicator.duplicate_object(
        distill_data,
        labels=preds,
        copy=True,
        task=Task.REGRESSION,
    )
    distill_model.fit(distill_data_with_labels, **fit_kw)


class TLearner(BaseSingleKPILearner):
  """Implementation of T-Learner uplift modelling.

    This model estimates the following uplift_score, given a set of
    features, X, and a treatment which can either be applied (T=1), or not
    applied (T=0):

      uplift_score = E[Y | T=1, X] - E[Y | T=0, X]

    This score can be interpreted as the absolute uplift of Y due to the
    treatment. Y can be set to either maximize_kpi, constraint_kpi or
    constraint_offset_kpi, depending on the kpi you want to target.
    It defaults to maximize_kpi.

  Attributes:
    base_model_control: The base ML model fitted on the control group.
    base_model_treatment: The base ML model fitted on the treatment group.
    target_kpi: The KPI type being targeted by this model.
  """

  def _setup_base_models(self, base_model: BaseModel):
    """Initialises the base models."""
    self.base_model_control = base_model
    self.base_model_treatment = object_duplicator.duplicate_object(base_model)

  def _fit(self, data: TrainData) -> None:
    """Fits the base models to the data.

    Args:
      data: The data to train the models on.

    Raises:
      ValueError: If the data does not have the target kpi.
    """
    data_has_kpi = (
        data.get_data_for_single_treatment(self.target_kpi, is_treated=False)
        is not None
    )
    if not data_has_kpi:
      raise ValueError(f"target_kpi {self.target_kpi.name} not in data")

    self.base_model_control.fit(
        data.get_data_for_single_treatment(self.target_kpi, is_treated=False)
    )
    self.base_model_treatment.fit(
        data.get_data_for_single_treatment(self.target_kpi, is_treated=True)
    )

  def _predict(self, data: Dataset) -> np.ndarray:
    """Predicts the uplift score.

    This predicts the uplift score. When comparing two uplift scores, you
    should target higher uplift scores with your treatment. This score can be
    interpreted as the estimated relative uplift the treatment has on the
    target KPI.

    It is calulated as:

    uplift_score = preds_treatment - preds_control

    Args:
      data: The features to predict on.

    Returns:
      The uplift score. The higher the score the higher the expected lift.
    """
    preds_control = self.base_model_control.predict(data)
    preds_treatment = self.base_model_treatment.predict(data)
    return preds_treatment - preds_control


class RetrospectiveLearner(BaseSingleKPILearner):
  """A class to implement retrospective estimation uplift modelling.

  This model estimates the following uplift_score, given a set of
  features, X, and a treatment which can either be applied (T=1), or not
  applied (T=0):

    uplift_score = E[Y | T=1, X] / E[Y | T=0, X]

  This score can be interpreted as the relative uplift of Y due to the
  treatment. Y can be any of the kpis: maximize_kpi, constraint_kpi or
  constraint_offset_kpi.

  This model is based on https://arxiv.org/abs/2008.06293

  Attributes:
    base_model: The model to predict the incrementality of the maximize_kpi kpi
  """

  def _setup_base_models(self, base_model: BaseModel):
    """Initialises the base models."""
    self.base_model = base_model

  def _fit(self, data: TrainData) -> None:
    """Fits the base model to the data.

    Args:
      data: The data to train the model on.
    """
    if not data.has_kpi(self.target_kpi):
      raise ValueError(f"target_kpi {self.target_kpi.name} not in data")
    if not data.has_non_negative_kpi(self.target_kpi):
      raise ValueError(f"target_kpi {self.target_kpi.name} must be >= 0")

    self.base_model.fit(data.get_inc_classifier_data(self.target_kpi))

  def _predict(self, data: Dataset) -> np.ndarray:
    """Predicts the uplift score.

    This predicts the uplift score. When comparing two uplift scores, you
    should target higher uplift scores with your treatment. This score can be
    interpreted as the estimated relative uplift the treatment has on the
    target KPI.

    It is calulated as:

    uplift_score = maximize_kpi_inc / (1 - maximize_kpi_inc)

    Args:
      data: The features to predict on.

    Returns:
      The uplift score. Higher scores mean the customer is better to target.
    """
    preds = self.base_model.predict(data)
    return preds / (1.0 - preds)


class FractionalRetrospectiveLearner(BaseFractionalLearner):
  """A class to implement fractional uplift modelling."""

  DEFAULT_CONSTRAINT_OFFSET_KPI_INC = 1.0
  DEFAULT_CONSTRAINT_OFFSET_KPI_WEIGHT = 0.0
  DEFAULT_CONSTRAINT_OFFSET_SCALE = 1.0

  def __init__(
      self,
      maximize_kpi_inc_model: BaseModel,
      constraint_kpi_inc_model: BaseModel | None = None,
      constraint_offset_kpi_inc_model: BaseModel | None = None,
      kpi_weight_model: BaseModel | None = None,
  ):
    """A class to implement fractional uplift modelling.

    This model estimates the following uplift_score, given a set of
    features, X, and a treatment which can either be applied (T=1), or not
    applied (T=0):

    uplift_score(X) = (
        (E[maximize_kpi | T=1, X] - E[maximize_kpi | T=0, X])
        / (
            E[constraint_kpi | T=1, X] - E[constraint_kpi | T=0, X]
            - (E[constraint_offset_kpi | T=1, X] - E[constraint_offset_kpi |
            T=0, X]) / constraint_offset_scale
        )
    )

    Maximize_kpi, constraint_kpi and constraint_offset_kpi are specified in the
    TrainData object which is
    passed
    to the fit() method. This will define what metrics the model will optimise
    for.

    The constraint_offset_scale is a constant, and so is only set at prediction
    time, and is only
    needed if the constraint_offset_kpi kpi is set.

    If constraint_offset_kpi is not specified in the train data, the model
    learns the following
    simplified uplift score:

    uplift_score(X) = (
        (E[maximize_kpi | T=1, X] - E[maximize_kpi | T=0, X])
        / (E[constraint_kpi | T=1, X] - E[constraint_kpi | T=0, X])
    )

    In this case, constraint_offset_scale is not needed.

    This can use any base models, and fit/predict on any dataset.

    Must provide the maximize_kpi_inc_model and the kpi_weight_model, but
    the
    other models only need to be provided if the dataset requires that model.

    Args:
      maximize_kpi_inc_model: The model to predict the incrementality of the
        maximize_kpi kpi (always needed)
      constraint_kpi_inc_model: The model to predict the incrementality of the
        constraint_kpi kpi (if not set will use the maximize_kpi_inc_model)
      constraint_offset_kpi_inc_model: The model to predict the incrementality
        of the constraint_offset_kpi kpi (if not set will use the
        maximize_kpi_inc_model)
      kpi_weight_model: The model to predict the relative weights of the
        maximize_kpi, constraint_kpi, constraint_offset_kpi kpis (if not set
        will use the maximize_kpi_inc_model)
    """
    super().__init__()
    self.maximize_kpi_inc_model = maximize_kpi_inc_model

    if constraint_kpi_inc_model is None:
      print(
          "constraint_kpi_inc_model not set, duplicating"
          " maximize_kpi_inc_model."
      )
      self.constraint_kpi_inc_model = object_duplicator.duplicate_object(
          maximize_kpi_inc_model
      )
    else:
      self.constraint_kpi_inc_model = constraint_kpi_inc_model

    if constraint_offset_kpi_inc_model is None:
      print(
          "constraint_offset_kpi_inc_model not set, duplicating"
          " maximize_kpi_inc_model."
      )
      self.constraint_offset_kpi_inc_model = object_duplicator.duplicate_object(
          maximize_kpi_inc_model
      )
    else:
      self.constraint_offset_kpi_inc_model = constraint_offset_kpi_inc_model

    if kpi_weight_model is None:
      print("kpi_weight_model not set, duplicating maximize_kpi_inc_model.")
      self.kpi_weight_model = object_duplicator.duplicate_object(
          maximize_kpi_inc_model
      )
    else:
      self.kpi_weight_model = kpi_weight_model

  def _validate_data(self, data: TrainData) -> None:
    """Validates that the data can be used for a fractional uplift model.

    The data must have at least the maximize_kpi and constraint_kpi kpis
    set. It can
    optionally also have the constraint_offset_kpi kpi.

    Args:
      data: The data to validate against.

    Raises:
      ValueError: If the maximize_kpi or constraint_kpi kpis are missing.
    """
    missing_maximize_kpi = not data.has_kpi(KPI.MAXIMIZE_KPI)
    missing_constraint_kpi = not data.has_kpi(KPI.CONSTRAINT_KPI)

    if missing_maximize_kpi | missing_constraint_kpi:
      raise ValueError(
          "FractionalRetrospectiveLearner requires maximize_kpi and"
          " constraint_kpi to be set in the train data. missing_maximize_kpi ="
          f" {missing_maximize_kpi}, missing_constraint_kpi ="
          f" {missing_constraint_kpi}."
      )

    if not data.has_non_negative_kpi(KPI.MAXIMIZE_KPI):
      raise ValueError(
          "maximize_kpi must be >= 0 for the FractionalRetrospectiveLearner"
      )

    if not data.has_non_negative_kpi(KPI.CONSTRAINT_KPI):
      raise ValueError(
          "constraint_kpi must be >= 0 for the FractionalRetrospectiveLearner"
      )

    has_negative_constraint_offset_kpi = data.has_kpi(
        KPI.CONSTRAINT_OFFSET_KPI
    ) & (not data.has_non_negative_kpi(KPI.CONSTRAINT_OFFSET_KPI))

    if has_negative_constraint_offset_kpi:
      raise ValueError(
          "constraint_offset_kpi must be >= 0 for the"
          "FractionalRetrospectiveLearner"
      )

  def _fit(self, data: TrainData) -> None:
    """Fits all of the required models.

    Args:
      data: The data to train the models on.
    """
    self._validate_data(data)

    self.maximize_kpi_inc_model.fit(
        data.get_inc_classifier_data(KPI.MAXIMIZE_KPI)
    )
    self.constraint_kpi_inc_model.fit(
        data.get_inc_classifier_data(KPI.CONSTRAINT_KPI)
    )
    self.kpi_weight_model.fit(data.get_kpi_weight_data())

    if data.has_kpi(KPI.CONSTRAINT_OFFSET_KPI):
      self.constraint_offset_kpi_inc_model.fit(
          data.get_inc_classifier_data(KPI.CONSTRAINT_OFFSET_KPI)
      )

  def _predict(
      self, data: Dataset, constraint_offset_scale: float | None = None
  ) -> np.ndarray:
    """Predicts the uplift score.

    This predicts the uplift score. When comparing two uplift scores, you
    should target higher uplift scores with your treatment.

    It is calulated as:

    uplift_score = (
        maximize_kpi_inc * maximize_kpi_weight
        / (constraint_kpi_inc * constraint_kpi_weight -
        (constraint_offset_kpi_inc * constraint_offset_kpi_weight) /
        constraint_offset_scale)
    )

    Args:
      data: The features to predict on
      constraint_offset_scale: The constraint_offset_scale (which is always
        fixed). Not needed if constraint_offset_kpi kpi is not provided.

    Returns:
      The uplift score. Higher scores mean the customer is better to target.

    Raises:
      RuntimeError: If the shape of the weights prediction is unexpected.
      ValueError: If constraint_offset_scale is not specified but the
      constraint_offset_kpi
      kpi exists.
    """

    # predict the kpi weights
    weights = self.kpi_weight_model.predict(data)

    if self.constraint_offset_kpi_inc_model.fitted:
      assert np.shape(weights) == (len(data), 3), (
          "The train data had a constraint_offset_kpi kpi, so the shape"
          f" of the weights should be ({len(data)}, 3). Instead got"
          f" {np.shape(weights)}."
      )
      if constraint_offset_scale is None:
        raise ValueError(
            "The train data had a constraint_offset_kpi kpi,"
            " constraint_offset_scale must not be None."
        )
    else:
      assert np.shape(weights) == (len(data),), (
          "The train data did not have a constraint_offset_kpi kpi, so"
          f" the shape of the weights should be ({len(data)},). Instead got"
          f" {np.shape(weights)}."
      )
      if constraint_offset_scale is not None:
        raise ValueError(
            "The train data did not have constraint_offset_kpi kpi,"
            " constraint_offset_scale must be None."
        )

    # predict the incrementality of maximize_kpi and constraint_kpi
    maximize_kpi_inc = 2.0 * self.maximize_kpi_inc_model.predict(data) - 1.0
    constraint_kpi_inc = 2.0 * self.constraint_kpi_inc_model.predict(data) - 1.0

    if self.constraint_offset_kpi_inc_model.fitted:
      # There are three scores per sample, split them out into the three
      # kpis
      maximize_kpi_weight = weights[:, KPI.MAXIMIZE_KPI.value]
      constraint_kpi_weight = weights[:, KPI.CONSTRAINT_KPI.value]
      constraint_offset_kpi_weight = weights[:, KPI.CONSTRAINT_OFFSET_KPI.value]

      # Since the constraint_offset_kpi kpi exists, we also need its incrementality
      constraint_offset_kpi_inc = (
          2.0 * self.constraint_offset_kpi_inc_model.predict(data) - 1.0
      )
    else:
      # There is only 1 score per sample, meaning this is a binomial
      # prediction. Assume there was no constraint_offset_kpi kpi, and it is the
      # probability for p_constraint_kpi
      constraint_kpi_weight = weights
      maximize_kpi_weight = 1.0 - constraint_kpi_weight

      constraint_offset_kpi_weight = self.DEFAULT_CONSTRAINT_OFFSET_KPI_WEIGHT
      constraint_offset_kpi_inc = self.DEFAULT_CONSTRAINT_OFFSET_KPI_INC
      constraint_offset_scale = self.DEFAULT_CONSTRAINT_OFFSET_SCALE

    out = (
        maximize_kpi_inc
        * maximize_kpi_weight
        / (
            constraint_kpi_inc * constraint_kpi_weight
            - constraint_offset_kpi_inc
            * constraint_offset_kpi_weight
            / constraint_offset_scale
        )
    )
    is_inf = (
        constraint_kpi_inc * constraint_kpi_weight
        < constraint_offset_kpi_inc
        * constraint_offset_kpi_weight
        / constraint_offset_scale
    )
    out[is_inf] = np.inf

    return out


class FractionalLearner(BaseFractionalLearner):
  """Implements fractional uplift modelling."""

  DEFAULT_CONSTRAINT_OFFSET_KPI_CATE = 0.0
  DEFAULT_CONSTRAINT_OFFSET_SCALE = 1.0

  def __init__(
      self,
      maximize_kpi_base_model: BaseModel,
      constraint_kpi_base_model: Optional[BaseModel] = None,
      constraint_offset_kpi_base_model: Optional[BaseModel] = None,
      single_kpi_learner: Type[BaseSingleKPILearner] = TLearner,
  ):
    """Implements fractional uplift modelling.

    This model estimates the following uplift_score, given a set of
    features, X, and a treatment which can either be applied (T=1), or not
    applied (T=0):

    uplift_score(X) = (
        (E[maximize_kpi | T=1, X] - E[maximize_kpi | T=0, X])
        / (
            E[constraint_kpi | T=1, X] - E[constraint_kpi | T=0, X]
            - (E[constraint_offset_kpi | T=1, X] - E[constraint_offset_kpi |
            T=0, X]) / constraint_offset_scale
        )
    )

    Maximize_kpi, constraint_kpi and constraint_offset_kpi are specified in the
    TrainData object which is
    passed
    to the fit() method. This will define what metrics the model will optimise
    for.

    The constraint_offset_scale is a constant, and so is only set at prediction
    time, and is only
    needed if the constraint_offset_kpi kpi is set.

    If constraint_offset_kpi is not specified in the train data, the model
    learns the following
    simplified uplift score:

    uplift_score(X) = (
        (E[maximize_kpi | T=1, X] - E[maximize_kpi | T=0, X])
        / (E[constraint_kpi | T=1, X] - E[constraint_kpi | T=0, X])
    )

    In this case, constraint_offset_scale is not needed at prediction time.

    This can use any base models, and fit/predict on any dataset.

    Must provide the maximize_kpi_inc_model and the kpi_weight_model, but
    the
    other models only need to be provided if the dataset requires that model.

    Args:
      maximize_kpi_base_model: The model to predict the incrementality of the
        maximize_kpi kpi (always needed)
      constraint_kpi_base_model: The model to predict the relative weights of
        the constraint_kpi kpi (if not set, will use the
        maximize_kpi_base_model)
      constraint_offset_kpi_base_model: The model to predict the incrementality
        of the constraint_offset_kpi kpi (if not set, will use the
        maximize_kpi_base_model)
      single_kpi_learner: The meta learner to use to learn the CATE for each
        kpi, defaults to a TLearner.
    """
    super().__init__()
    self.maximize_kpi_cate_learner = single_kpi_learner(
        base_model=maximize_kpi_base_model,
        target_kpi=KPI.MAXIMIZE_KPI,
    )
    self.constraint_kpi_cate_learner = single_kpi_learner(
        base_model=constraint_kpi_base_model
        or object_duplicator.duplicate_object(maximize_kpi_base_model),
        target_kpi=KPI.CONSTRAINT_KPI,
    )
    self.constraint_offset_kpi_cate_learner = single_kpi_learner(
        base_model=constraint_offset_kpi_base_model
        or object_duplicator.duplicate_object(maximize_kpi_base_model),
        target_kpi=KPI.CONSTRAINT_OFFSET_KPI,
    )

  def _validate_data(self, data: TrainData) -> None:
    """Validates that the data can be used for a fractional uplift model.

    The data must have at least the maximize_kpi and constraint_kpi kpis
    set. It can
    optionally also have the constraint_offset_kpi kpi.

    Args:
      data: The data to validate against.

    Raises:
      ValueError: If the maximize_kpi or constraint_kpi kpis are missing.
    """
    missing_maximize_kpi = not data.has_kpi(KPI.MAXIMIZE_KPI)
    missing_constraint_kpi = not data.has_kpi(KPI.CONSTRAINT_KPI)

    if missing_maximize_kpi | missing_constraint_kpi:
      raise ValueError(
          "FractionalRetrospectiveLearner requires maximize_kpi and"
          " constraint_kpi to be set in the train data. missing_maximize_kpi ="
          f" {missing_maximize_kpi}, missing_constraint_kpi ="
          f" {missing_constraint_kpi}."
      )

  def _fit(self, data: TrainData) -> None:
    """Fits all of the required models.

    Args:
      data: The data to train the models on.
    """
    self._validate_data(data)

    self.maximize_kpi_cate_learner.fit(data)
    self.constraint_kpi_cate_learner.fit(data)

    if data.has_kpi(KPI.CONSTRAINT_OFFSET_KPI):
      self.constraint_offset_kpi_cate_learner.fit(data)

  def _predict(
      self, data: Dataset, constraint_offset_scale: Optional[float] = None
  ) -> np.ndarray:
    """Predicts the uplift score.

    This predicts the uplift score. When comparing two uplift scores, you
    should target higher uplift scores with your treatment.

    It is calulated as:

    uplift_score = (
        maximize_kpi_cate
        / (constraint_kpi_cate - constraint_offset_kpi_cate /
        constraint_offset_scale)
    )

    Args:
      data: The features to predict on
      constraint_offset_scale: The constraint_offset_scale kpi (which is a
        constant). Not needed if constraint_offset_kpi kpi is not provided.

    Returns:
      The uplift score. Higher scores mean the customer is better to target.

    Raises:
      ValueError: If constraint_offset_scale is specified but the
      constraint_offset_kpi kpi
      does not exist,
      or vica versa.
    """
    if (constraint_offset_scale is not None) & (
        not self.constraint_offset_kpi_cate_learner.fitted
    ):
      raise ValueError(
          "This learner was trained on data without a constraint_offset_kpi"
          " kpi, cannot set constraint_offset_scale when predicting"
      )

    if (constraint_offset_scale is None) & (
        self.constraint_offset_kpi_cate_learner.fitted
    ):
      raise ValueError(
          "This learner was trained on data with a constraint_offset_kpi"
          " kpi, must set constraint_offset_scale when predicting"
      )

    maximize_kpi_cate_pred = self.maximize_kpi_cate_learner.predict(data)
    constraint_kpi_cate_pred = self.constraint_kpi_cate_learner.predict(data)

    if constraint_offset_scale is not None:
      constraint_offset_kpi_cate_pred = (
          self.constraint_offset_kpi_cate_learner.predict(data)
      )
    else:
      constraint_offset_kpi_cate_pred = self.DEFAULT_CONSTRAINT_OFFSET_KPI_CATE
      constraint_offset_scale = self.DEFAULT_CONSTRAINT_OFFSET_SCALE

    out = maximize_kpi_cate_pred / (
        constraint_kpi_cate_pred
        - constraint_offset_kpi_cate_pred / constraint_offset_scale
    )
    is_inf = (
        constraint_kpi_cate_pred
        < constraint_offset_kpi_cate_pred / constraint_offset_scale
    )
    out[is_inf] = np.inf
    return out
