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

"""Tests for the uplift model classes."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from fractional_uplift import base_models
from fractional_uplift import constants
from fractional_uplift import datasets
from fractional_uplift import meta_learners


KPI = constants.KPI


def make_mock_model(preds):
  """Makes a mock base model that returns preds when predict is called."""
  model = mock.MagicMock(spec=base_models.TensorflowDecisionForestClassifier)
  model.predict.return_value = preds
  model.fitted = False

  def mock_fit(self):
    self.fitted = True

  model.fit.side_effect = lambda *args, **kwargs: mock_fit(model)
  return model


def make_mock_single_kpi_learner(base_model, target_kpi):
  """Makes a mock single kpi meta learner.

  It will return the predictions from the base model when predict is called.
  """
  model = mock.MagicMock(spec=meta_learners.TLearner)
  model.predict.return_value = base_model.predict.return_value
  model.fitted = False
  model.target_kpi = target_kpi

  def mock_fit(self):
    self.fitted = True

  model.fit.side_effect = lambda *args, **kwargs: mock_fit(model)
  return model


class FractionalRetrospectiveLearnerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    self.maximize_kpi_inc_preds = np.array([0.1, 0.5, 0.3, 0.8])
    self.constraint_kpi_inc_preds = np.array([0.3, 0.3, 0.1, 0.05])
    self.constraint_offset_kpi_inc_preds = np.array([0.6, 0.2, 0.6, 0.15])
    self.maximize_kpi_weight_preds = np.array([0.1, 0.5, 0.3, 0.8])
    self.constraint_kpi_weight_preds = np.array([0.3, 0.3, 0.1, 0.05])
    self.constraint_offset_kpi_weight_preds = np.array([0.6, 0.2, 0.6, 0.15])
    self.kpi_weight_preds = np.concatenate(
        [
            self.maximize_kpi_weight_preds.reshape(-1, 1),
            self.constraint_kpi_weight_preds.reshape(-1, 1),
            self.constraint_offset_kpi_weight_preds.reshape(-1, 1),
        ],
        axis=1,
    )

    self.maximize_kpi_inc_data = mock.MagicMock(spec=datasets.PandasDataset)
    self.constraint_kpi_inc_data = mock.MagicMock(spec=datasets.PandasDataset)
    self.constraint_offset_kpi_inc_data = mock.MagicMock(
        spec=datasets.PandasDataset
    )
    self.kpi_weight_data = mock.MagicMock(spec=datasets.PandasDataset)

  def test_model_with_maximize_constraint_and_constraint_offset_kpis(
      self,
  ):
    # Tests the case where:
    # - All three KPIs exist.
    # - All three KPIs require incrementality models.

    maximize_kpi_inc_model = make_mock_model(self.maximize_kpi_inc_preds)
    constraint_kpi_inc_model = make_mock_model(self.constraint_kpi_inc_preds)
    constraint_offset_kpi_inc_model = make_mock_model(
        self.constraint_offset_kpi_inc_preds
    )
    kpi_weight_model = make_mock_model(self.kpi_weight_preds)

    inc_data = {
        KPI.MAXIMIZE_KPI: self.maximize_kpi_inc_data,
        KPI.CONSTRAINT_KPI: self.constraint_kpi_inc_data,
        KPI.CONSTRAINT_OFFSET_KPI: self.constraint_offset_kpi_inc_data,
    }
    train_data = mock.MagicMock(spec=datasets.PandasTrainData)
    train_data.get_inc_classifier_data = inc_data.get
    train_data.get_kpi_weight_data.return_value = self.kpi_weight_data
    train_data.has_kpi = lambda c: inc_data.get(c) is not None
    train_data.has_non_negative_kpi = lambda c: inc_data.get(c) is not None

    model = meta_learners.FractionalRetrospectiveLearner(
        maximize_kpi_inc_model=maximize_kpi_inc_model,
        constraint_kpi_inc_model=constraint_kpi_inc_model,
        constraint_offset_kpi_inc_model=constraint_offset_kpi_inc_model,
        kpi_weight_model=kpi_weight_model,
    )

    self.assertFalse(model.fitted)
    model.fit(train_data)
    self.assertTrue(model.fitted)

    maximize_kpi_inc_model.fit.assert_called_with(self.maximize_kpi_inc_data)
    constraint_kpi_inc_model.fit.assert_called_with(
        self.constraint_kpi_inc_data
    )
    constraint_offset_kpi_inc_model.fit.assert_called_with(
        self.constraint_offset_kpi_inc_data
    )
    kpi_weight_model.fit.assert_called_with(self.kpi_weight_data)

    constraint_offset_scale = 0.5
    pred_data = mock.MagicMock()
    pred_data.__len__.return_value = 4
    actual_output = model.predict(
        pred_data, constraint_offset_scale=constraint_offset_scale
    )

    maximize_kpi_inc_model.predict.assert_called_with(pred_data)
    constraint_kpi_inc_model.predict.assert_called_with(pred_data)
    constraint_offset_kpi_inc_model.predict.assert_called_with(pred_data)
    kpi_weight_model.predict.assert_called_with(pred_data)

    # Calculate expected output
    maximize_kpi_cate = (
        2.0 * self.maximize_kpi_inc_preds - 1
    ) * self.maximize_kpi_weight_preds
    constraint_kpi_cate = (
        2.0 * self.constraint_kpi_inc_preds - 1
    ) * self.constraint_kpi_weight_preds
    constraint_offset_kpi_cate = (
        2.0 * self.constraint_offset_kpi_inc_preds - 1
    ) * self.constraint_offset_kpi_weight_preds
    expected_output = maximize_kpi_cate / (
        constraint_kpi_cate
        - constraint_offset_kpi_cate / constraint_offset_scale
    )
    is_inf = (
        constraint_offset_kpi_cate / constraint_offset_scale
        >= constraint_kpi_cate
    )
    expected_output[is_inf] = np.inf

    np.testing.assert_array_almost_equal(
        actual_output, expected_output.flatten()
    )

  def test_model_with_maximize_and_constraint_kpis(self):
    # Tests the case where:
    # - Only the maximize_kpi and constraint_kpi exist.
    # - Both KPIs need an incrementality model.

    maximize_kpi_inc_model = make_mock_model(self.maximize_kpi_inc_preds)
    constraint_kpi_inc_model = make_mock_model(self.constraint_kpi_inc_preds)
    constraint_offset_kpi_inc_model = make_mock_model(
        self.constraint_offset_kpi_inc_preds
    )

    # When there are only two KPIs, the kpi weight model only
    # returns the constraint_kpi weight, because it's a binomial classifier
    kpi_weight_model = make_mock_model(self.constraint_kpi_weight_preds)

    inc_data = {
        KPI.MAXIMIZE_KPI: self.maximize_kpi_inc_data,
        KPI.CONSTRAINT_KPI: self.constraint_kpi_inc_data,
    }
    train_data = mock.MagicMock(spec=datasets.PandasTrainData)
    train_data.get_inc_classifier_data = inc_data.get
    train_data.get_kpi_weight_data.return_value = self.kpi_weight_data
    train_data.has_kpi = lambda c: inc_data.get(c) is not None
    train_data.has_non_negative_kpi = lambda c: inc_data.get(c) is not None

    model = meta_learners.FractionalRetrospectiveLearner(
        maximize_kpi_inc_model=maximize_kpi_inc_model,
        constraint_kpi_inc_model=constraint_kpi_inc_model,
        constraint_offset_kpi_inc_model=constraint_offset_kpi_inc_model,
        kpi_weight_model=kpi_weight_model,
    )

    self.assertFalse(model.fitted)
    model.fit(train_data)
    self.assertTrue(model.fitted)

    maximize_kpi_inc_model.fit.assert_called_with(self.maximize_kpi_inc_data)
    constraint_kpi_inc_model.fit.assert_called_with(
        self.constraint_kpi_inc_data
    )
    constraint_offset_kpi_inc_model.fit.assert_not_called()
    kpi_weight_model.fit.assert_called_with(self.kpi_weight_data)

    pred_data = mock.MagicMock()
    pred_data.__len__.return_value = 4
    actual_output = model.predict(pred_data, constraint_offset_scale=None)

    maximize_kpi_inc_model.predict.assert_called_with(pred_data)
    constraint_kpi_inc_model.predict.assert_called_with(pred_data)
    constraint_offset_kpi_inc_model.predict.assert_not_called()
    kpi_weight_model.predict.assert_called_with(pred_data)

    # Calculate expected output
    maximize_kpi_cate = (2.0 * self.maximize_kpi_inc_preds - 1) * (
        1.0 - self.constraint_kpi_weight_preds
    )
    constraint_kpi_cate = (
        2.0 * self.constraint_kpi_inc_preds - 1
    ) * self.constraint_kpi_weight_preds
    expected_output = maximize_kpi_cate / constraint_kpi_cate
    is_inf = 0.0 >= constraint_kpi_cate
    expected_output[is_inf] = np.inf

    np.testing.assert_array_almost_equal(
        actual_output, expected_output.flatten()
    )

  def test_model_raises_exception_when_constraint_offset_scale_is_set_but_not_needed(
      self,
  ):
    inc_data = {
        KPI.MAXIMIZE_KPI: self.maximize_kpi_inc_data,
        KPI.CONSTRAINT_KPI: self.constraint_kpi_inc_data,
    }
    train_data = mock.MagicMock(spec=datasets.PandasTrainData)
    train_data.get_kpi_weight_data.return_value = mock.MagicMock()
    train_data.get_inc_classifier_data = inc_data.get
    train_data.has_kpi = lambda c: inc_data.get(c) is not None
    train_data.has_non_negative_kpi = lambda c: inc_data.get(c) is not None

    model = meta_learners.FractionalRetrospectiveLearner(
        maximize_kpi_inc_model=make_mock_model(self.maximize_kpi_inc_preds),
        constraint_kpi_inc_model=make_mock_model(self.constraint_kpi_inc_preds),
        constraint_offset_kpi_inc_model=make_mock_model(
            self.constraint_offset_kpi_inc_preds
        ),
        kpi_weight_model=make_mock_model(self.constraint_kpi_weight_preds),
    )
    model.fit(train_data)
    prediction_data = mock.MagicMock()
    prediction_data.__len__.return_value = 4

    with self.assertRaises(ValueError):
      model.predict(prediction_data, constraint_offset_scale=1.0)

  def test_model_raises_exception_when_constraint_offset_scale_not_set_but_needed(
      self,
  ):
    inc_data = {
        KPI.MAXIMIZE_KPI: self.maximize_kpi_inc_data,
        KPI.CONSTRAINT_KPI: self.constraint_kpi_inc_data,
        KPI.CONSTRAINT_OFFSET_KPI: self.constraint_offset_kpi_inc_data,
    }
    train_data = mock.MagicMock(spec=datasets.PandasTrainData)
    train_data.get_kpi_weight_data.return_value = mock.MagicMock()
    train_data.get_inc_classifier_data = inc_data.get
    train_data.has_kpi = lambda c: inc_data.get(c) is not None
    train_data.has_non_negative_kpi = lambda c: inc_data.get(c) is not None

    pred_data = mock.MagicMock()
    pred_data.__len__.return_value = 4

    model = meta_learners.FractionalRetrospectiveLearner(
        maximize_kpi_inc_model=make_mock_model(self.maximize_kpi_inc_preds),
        constraint_kpi_inc_model=make_mock_model(self.constraint_kpi_inc_preds),
        constraint_offset_kpi_inc_model=make_mock_model(
            self.constraint_offset_kpi_inc_preds
        ),
        kpi_weight_model=make_mock_model(self.kpi_weight_preds),
    )
    model.fit(train_data)
    with self.assertRaises(ValueError):
      model.predict(pred_data)

  @parameterized.parameters(list(KPI))
  def test_model_raises_exception_if_any_kpi_is_negative(self, bad_kpi):
    inc_data = {
        KPI.MAXIMIZE_KPI: self.maximize_kpi_inc_data,
        KPI.CONSTRAINT_KPI: self.constraint_kpi_inc_data,
        KPI.CONSTRAINT_OFFSET_KPI: self.constraint_offset_kpi_inc_data,
    }

    train_data = mock.MagicMock(spec=datasets.PandasTrainData)
    train_data.get_kpi_weight_data.return_value = mock.MagicMock()
    train_data.get_inc_classifier_data = inc_data.get
    train_data.has_kpi = lambda c: inc_data.get(c) is not None
    train_data.has_non_negative_kpi = lambda c: c != bad_kpi

    model = meta_learners.FractionalRetrospectiveLearner(
        maximize_kpi_inc_model=make_mock_model(self.maximize_kpi_inc_preds),
        constraint_kpi_inc_model=make_mock_model(self.constraint_kpi_inc_preds),
        constraint_offset_kpi_inc_model=make_mock_model(
            self.constraint_offset_kpi_inc_preds
        ),
        kpi_weight_model=make_mock_model(self.kpi_weight_preds),
    )
    with self.assertRaises(ValueError):
      model.fit(train_data)


class RetrospectiveLearnerTest(parameterized.TestCase):

  @parameterized.parameters(list(KPI))
  def test_model_fits_and_predicts_correctly(self, target_kpi):
    inc_preds = np.array([0.1, 0.5, 0.3, 0.8])
    inc_data = {
        KPI.MAXIMIZE_KPI: mock.MagicMock(spec=datasets.PandasDataset),
        KPI.CONSTRAINT_KPI: mock.MagicMock(spec=datasets.PandasDataset),
        KPI.CONSTRAINT_OFFSET_KPI: mock.MagicMock(spec=datasets.PandasDataset),
    }
    train_data = mock.MagicMock(spec=datasets.PandasTrainData)
    train_data.get_kpi_weight_data.return_value = None
    train_data.get_inc_classifier_data = inc_data.get
    train_data.has_kpi = lambda c: inc_data.get(c) is not None
    train_data.has_non_negative_kpi = lambda c: inc_data.get(c) is not None

    base_model = make_mock_model(preds=inc_preds)
    model = meta_learners.RetrospectiveLearner(base_model, target_kpi)

    self.assertFalse(model.fitted)
    model.fit(train_data)
    self.assertTrue(model.fitted)

    base_model.fit.assert_called_with(inc_data[target_kpi])

    pred_data = mock.MagicMock()
    pred_data.__len__.return_value = 4
    actual_output = model.predict(pred_data)

    base_model.predict.assert_called_with(pred_data)

    # Calculate expected output
    expected_output = inc_preds / (1.0 - inc_preds)

    np.testing.assert_array_almost_equal(
        actual_output, expected_output
    )

  @parameterized.parameters([KPI.CONSTRAINT_KPI, KPI.CONSTRAINT_OFFSET_KPI])
  def test_model_raises_exception_when_data_is_missing_for_target_kpi(
      self, target_kpi
  ):
    inc_preds = np.array([0.1, 0.5, 0.3, 0.8])
    inc_data = {
        KPI.MAXIMIZE_KPI: mock.MagicMock(spec=datasets.PandasDataset),
    }
    train_data = mock.MagicMock(spec=datasets.PandasTrainData)
    train_data.get_kpi_weight_data.return_value = None
    train_data.get_inc_classifier_data = inc_data.get
    train_data.has_kpi = lambda c: inc_data.get(c) is not None
    train_data.has_non_negative_kpi = lambda c: inc_data.get(c) is not None

    base_model = make_mock_model(preds=inc_preds)
    model = meta_learners.RetrospectiveLearner(base_model, target_kpi)

    with self.assertRaises(ValueError):
      model.fit(train_data)

  @parameterized.parameters([KPI.CONSTRAINT_KPI, KPI.CONSTRAINT_OFFSET_KPI])
  def test_model_raises_exception_when_data_is_negative_for_target_kpi(
      self, target_kpi
  ):
    inc_preds = np.array([0.1, 0.5, 0.3, 0.8])
    inc_data = {
        KPI.MAXIMIZE_KPI: mock.MagicMock(spec=datasets.PandasDataset),
    }
    train_data = mock.MagicMock(spec=datasets.PandasTrainData)
    train_data.get_kpi_weight_data.return_value = None
    train_data.get_inc_classifier_data = inc_data.get
    train_data.has_kpi = lambda c: inc_data.get(c) is not None
    train_data.has_non_negative_kpi.return_value = False

    base_model = make_mock_model(preds=inc_preds)
    model = meta_learners.RetrospectiveLearner(base_model, target_kpi)

    with self.assertRaises(ValueError):
      model.fit(train_data)

def prepare_mock_train_data(kpis):
  mock_datasets = {}
  for kpi in kpis:
    mock_datasets[kpi] = mock.MagicMock(spec=datasets.PandasDataset)
    mock_datasets[(kpi, False)] = mock.MagicMock(spec=datasets.PandasDataset)
    mock_datasets[(kpi, True)] = mock.MagicMock(spec=datasets.PandasDataset)

  if KPI.CONSTRAINT_KPI in kpis:
    mock_datasets["kpi_weight"] = mock.MagicMock(spec=datasets.PandasDataset)

  train_data = mock.MagicMock(spec=datasets.PandasTrainData)

  def _get_data_for_single_treatment(kpi, is_treated):
    return mock_datasets.get((kpi, is_treated))

  def _get_inc_classifier_data(kpi):
    return mock_datasets.get(kpi)

  def _get_kpi_weight_data():
    return mock_datasets.get("kpi_weight")

  train_data.get_data_for_single_treatment = _get_data_for_single_treatment
  train_data.get_inc_classifier_data = _get_inc_classifier_data
  train_data.get_kpi_weight_data = _get_kpi_weight_data
  train_data.has_kpi = lambda c: c in kpis

  return train_data, mock_datasets


class TLearnerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.preds_control = np.array([0.1, 0.5, 0.3, 0.8])
    self.preds_treatment = np.array([0.2, 2.5, -0.3, 1.3])

  @parameterized.parameters(list(KPI))
  def test_model_fits_and_predicts_correctly(self, target_kpi):
    train_data, inc_data = prepare_mock_train_data([
        KPI.MAXIMIZE_KPI,
        KPI.CONSTRAINT_KPI,
        KPI.CONSTRAINT_OFFSET_KPI,
    ])
    pred_data = mock.MagicMock()
    pred_data.__len__.return_value = 4
    base_model_control = make_mock_model(preds=self.preds_control)
    base_model_treatment = make_mock_model(preds=self.preds_treatment)

    with mock.patch.object(
        meta_learners.object_duplicator,
        "duplicate_object",
        return_value=base_model_treatment,
    ):
      model = meta_learners.TLearner(base_model_control, target_kpi)

      self.assertFalse(model.fitted)
      model.fit(train_data)
      self.assertTrue(model.fitted)
      actual_output = model.predict(pred_data)

    base_model_control.fit.assert_called_with(inc_data[(target_kpi, False)])
    base_model_treatment.fit.assert_called_with(inc_data[(target_kpi, True)])
    base_model_control.predict.assert_called_with(pred_data)
    base_model_treatment.predict.assert_called_with(pred_data)
    expected_output = (self.preds_treatment - self.preds_control).flatten()
    np.testing.assert_array_almost_equal(actual_output, expected_output)

  @parameterized.parameters([KPI.CONSTRAINT_KPI, KPI.CONSTRAINT_OFFSET_KPI])
  def test_model_raises_exception_when_data_is_missing_for_target_kpi(
      self, target_kpi
  ):
    train_data, _ = prepare_mock_train_data([KPI.MAXIMIZE_KPI])

    base_model_control = make_mock_model(preds=self.preds_control)
    base_model_treatment = make_mock_model(preds=self.preds_treatment)
    with mock.patch.object(
        meta_learners.object_duplicator,
        "duplicate_object",
        return_value=base_model_treatment,
    ):
      model = meta_learners.TLearner(base_model_control, target_kpi)

      with self.assertRaises(ValueError):
        model.fit(train_data)


class FractionalLearnerTest(parameterized.TestCase):
  def setUp(self):
    super().setUp()
    self.maximize_kpi_inc_preds = np.array([0.1, 0.5, 0.3, 0.8])
    self.constraint_kpi_inc_preds = np.array([1.2, 0.8, 0.2, 1.0])
    self.constraint_offset_kpi_inc_preds = np.array([0.11, 0.3, 0.2, 0.1])

    self.mock_maximize_kpi_base_model = make_mock_model(
        preds=self.maximize_kpi_inc_preds
    )
    self.mock_constraint_kpi_base_model = make_mock_model(
        preds=self.constraint_kpi_inc_preds
    )
    self.mock_constraint_offset_kpi_base_model = make_mock_model(
        preds=self.constraint_offset_kpi_inc_preds
    )

    self.constraint_offset_scale = 0.5

    self.expected_preds_with_constraint_offset_kpi = (
        self.maximize_kpi_inc_preds
        / (
            self.constraint_kpi_inc_preds
            - self.constraint_offset_kpi_inc_preds
            / self.constraint_offset_scale
        )
    )
    self.expected_preds_with_constraint_offset_kpi[
        self.constraint_kpi_inc_preds
        <= self.constraint_offset_kpi_inc_preds / self.constraint_offset_scale
    ] = np.inf
    self.expected_preds_with_constraint_offset_kpi = (
        self.expected_preds_with_constraint_offset_kpi.flatten()
    )

    self.expected_preds_without_constraint_offset_kpi = (
        self.maximize_kpi_inc_preds / self.constraint_kpi_inc_preds
    )
    self.expected_preds_without_constraint_offset_kpi[
        self.constraint_kpi_inc_preds <= 0.0
    ] = np.inf
    self.expected_preds_without_constraint_offset_kpi = (
        self.expected_preds_without_constraint_offset_kpi.flatten()
    )

  def test_single_kpi_learners_are_instantiated_with_correct_kpis(self):
    model = meta_learners.FractionalLearner(
        maximize_kpi_base_model=self.mock_maximize_kpi_base_model,
        constraint_kpi_base_model=self.mock_constraint_kpi_base_model,
        constraint_offset_kpi_base_model=self.mock_constraint_offset_kpi_base_model,
        single_kpi_learner=make_mock_single_kpi_learner,
    )

    self.assertEqual(
        model.maximize_kpi_cate_learner.target_kpi, KPI.MAXIMIZE_KPI
    )
    self.assertEqual(
        model.constraint_kpi_cate_learner.target_kpi,
        KPI.CONSTRAINT_KPI,
    )
    self.assertEqual(
        model.constraint_offset_kpi_cate_learner.target_kpi,
        KPI.CONSTRAINT_OFFSET_KPI,
    )

  def test_model_fits_all_learners_with_all_kpis(self):
    model = meta_learners.FractionalLearner(
        maximize_kpi_base_model=self.mock_maximize_kpi_base_model,
        constraint_kpi_base_model=self.mock_constraint_kpi_base_model,
        constraint_offset_kpi_base_model=self.mock_constraint_offset_kpi_base_model,
        single_kpi_learner=make_mock_single_kpi_learner,
    )
    train_data, _ = prepare_mock_train_data([
        KPI.MAXIMIZE_KPI,
        KPI.CONSTRAINT_KPI,
        KPI.CONSTRAINT_OFFSET_KPI,
    ])

    model.fit(train_data)

    model.maximize_kpi_cate_learner.fit.assert_called_with(train_data)
    model.constraint_kpi_cate_learner.fit.assert_called_with(train_data)
    model.constraint_offset_kpi_cate_learner.fit.assert_called_with(train_data)

  def test_model_does_not_fit_constraint_offset_kpi_learner_without_constraint_offset_kpi(
      self,
  ):
    model = meta_learners.FractionalLearner(
        maximize_kpi_base_model=self.mock_maximize_kpi_base_model,
        constraint_kpi_base_model=self.mock_constraint_kpi_base_model,
        constraint_offset_kpi_base_model=self.mock_constraint_offset_kpi_base_model,
        single_kpi_learner=make_mock_single_kpi_learner,
    )
    train_data, _ = prepare_mock_train_data(
        [KPI.MAXIMIZE_KPI, KPI.CONSTRAINT_KPI]
    )

    model.fit(train_data)

    model.maximize_kpi_cate_learner.fit.assert_called_with(train_data)
    model.constraint_kpi_cate_learner.fit.assert_called_with(train_data)
    model.constraint_offset_kpi_cate_learner.fit.assert_not_called()

  def test_model_combines_cate_predictions_correctly_with_all_kpis(self):
    train_data, _ = prepare_mock_train_data([
        KPI.MAXIMIZE_KPI,
        KPI.CONSTRAINT_KPI,
        KPI.CONSTRAINT_OFFSET_KPI,
    ])
    prediction_data = mock.MagicMock()
    model = meta_learners.FractionalLearner(
        maximize_kpi_base_model=self.mock_maximize_kpi_base_model,
        constraint_kpi_base_model=self.mock_constraint_kpi_base_model,
        constraint_offset_kpi_base_model=self.mock_constraint_offset_kpi_base_model,
        single_kpi_learner=make_mock_single_kpi_learner,
    )
    model.fit(train_data)

    preds = model.predict(
        prediction_data, constraint_offset_scale=self.constraint_offset_scale
    )

    model.maximize_kpi_cate_learner.predict.assert_called_with(prediction_data)
    model.constraint_kpi_cate_learner.predict.assert_called_with(
        prediction_data
    )
    model.constraint_offset_kpi_cate_learner.predict.assert_called_with(
        prediction_data
    )
    np.testing.assert_allclose(
        preds, self.expected_preds_with_constraint_offset_kpi
    )

  def test_model_combines_cate_predictions_correctly_without_constraint_offset_kpi(
      self,
  ):
    train_data, _ = prepare_mock_train_data(
        [KPI.MAXIMIZE_KPI, KPI.CONSTRAINT_KPI]
    )
    prediction_data = mock.MagicMock()
    model = meta_learners.FractionalLearner(
        maximize_kpi_base_model=self.mock_maximize_kpi_base_model,
        constraint_kpi_base_model=self.mock_constraint_kpi_base_model,
        constraint_offset_kpi_base_model=self.mock_constraint_offset_kpi_base_model,
        single_kpi_learner=make_mock_single_kpi_learner,
    )
    model.fit(train_data)

    preds = model.predict(prediction_data, constraint_offset_scale=None)

    model.maximize_kpi_cate_learner.predict.assert_called_with(prediction_data)
    model.constraint_kpi_cate_learner.predict.assert_called_with(
        prediction_data
    )
    model.constraint_offset_kpi_cate_learner.predict.assert_not_called()
    np.testing.assert_allclose(
        preds, self.expected_preds_without_constraint_offset_kpi
    )


if __name__ == "__main__":
  absltest.main()
