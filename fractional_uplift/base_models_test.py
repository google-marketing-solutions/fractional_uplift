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

"""Tests for all of the base models."""
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import pandas as pd
import tensorflow_decision_forests as tfdf
import inspect

from fractional_uplift import base_models
from fractional_uplift import datasets


BASE_CLASSIFIERS = [
    dict(
        testcase_name="TensorflowDecisionForestClassifier(Default)",
        base_model_cls=base_models.TensorflowDecisionForestClassifier,
        model_params={},
    ),
    dict(
        testcase_name="TensorflowDecisionForestClassifier(RandomForest)",
        base_model_cls=base_models.TensorflowDecisionForestClassifier,
        model_params=dict(decision_forest_cls=tfdf.keras.RandomForestModel),
    ),
]

BASE_REGRESSORS = [
    dict(
        testcase_name="TensorflowDecisionForestRegressor(Default)",
        base_model_cls=base_models.TensorflowDecisionForestRegressor,
        model_params={},
    ),
    dict(
        testcase_name="TensorflowDecisionForestRegressor(RandomForest)",
        base_model_cls=base_models.TensorflowDecisionForestRegressor,
        model_params=dict(decision_forest_cls=tfdf.keras.RandomForestModel),
    ),
]


class AllBaseModelsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    self.pd_dataframe = pd.DataFrame({
        "X1": [1.0, 2.0, 3.0],
        "X2": [1.0, 4.0, 3.0],
        "X3": pd.Categorical(["a", "b", "c"]),
    })
    self.binomial_data = datasets.PandasDataset(
        self.pd_dataframe.copy(),
        labels=np.array([0, 1, 1]),
        weights=np.array([2.0, 1.0, 1.0]),
        shuffle=False,
        task=tfdf.keras.Task.CLASSIFICATION,
    )
    self.multiclass_data = datasets.PandasDataset(
        self.pd_dataframe.copy(),
        labels=np.array([0, 1, 2]),
        weights=np.array([2.0, 1.0, 1.0]),
        shuffle=False,
        task=tfdf.keras.Task.CLASSIFICATION,
    )
    self.regression_data = datasets.PandasDataset(
        self.pd_dataframe.copy(),
        labels=np.array([0.0, 1.0, 2.2]),
        weights=np.array([2.0, 1.0, 1.0]),
        shuffle=False,
        task=tfdf.keras.Task.REGRESSION,
    )
    self.binomial_data_const = datasets.PandasDataset(
        self.pd_dataframe.copy(),
        labels=np.array([1, 1, 1]),
        weights=np.array([2.0, 1.0, 1.0]),
        shuffle=False,
        task=tfdf.keras.Task.CLASSIFICATION,
    )
    self.regression_data_const = datasets.PandasDataset(
        self.pd_dataframe.copy(),
        labels=np.array([0.5, 0.5, 0.5]),
        weights=np.array([2.0, 1.0, 1.0]),
        shuffle=False,
        task=tfdf.keras.Task.REGRESSION,
    )
    self.pred_data = datasets.PandasDataset(
        self.pd_dataframe.copy(), shuffle=False
    )

  @parameterized.named_parameters(BASE_REGRESSORS + BASE_CLASSIFIERS)
  def test_models_start_unfitted(self, base_model_cls, model_params):
    model = base_model_cls(**model_params)

    self.assertIsInstance(model, base_models.BaseModel)
    self.assertFalse(model.fitted)

  @parameterized.named_parameters(BASE_REGRESSORS)
  def test_regressors_can_fit(self, base_model_cls, model_params):
    model = base_model_cls(**model_params)
    model.fit(self.regression_data)

    self.assertTrue(model.fitted)

  @parameterized.named_parameters(BASE_CLASSIFIERS)
  def test_classifiers_can_fit_binomial_data(
      self, base_model_cls, model_params
  ):
    model = base_model_cls(**model_params)
    model.fit(self.binomial_data)

    self.assertTrue(model.fitted)

  @parameterized.named_parameters(BASE_CLASSIFIERS)
  def test_classifiers_can_fit_multiclass_data(
      self, base_model_cls, model_params
  ):
    model = base_model_cls(**model_params)
    model.fit(self.multiclass_data)

    self.assertTrue(model.fitted)

  @parameterized.named_parameters(BASE_REGRESSORS)
  def test_regressors_can_predict_if_fitted(self, base_model_cls, model_params):
    model = base_model_cls(**model_params)
    model.fit(self.regression_data)
    preds = model.predict(self.pred_data)

    self.assertIsInstance(preds, np.ndarray)
    self.assertEqual(preds.dtype, np.float32)
    self.assertEqual(np.shape(preds), (3,))

  @parameterized.named_parameters(BASE_CLASSIFIERS)
  def test_classifiers_can_predict_if_fitted_binomial_data(
      self, base_model_cls, model_params
  ):
    model = base_model_cls(**model_params)
    model.fit(self.binomial_data)
    preds = model.predict(self.pred_data)

    self.assertIsInstance(preds, np.ndarray)
    self.assertEqual(preds.dtype, np.float32)
    self.assertEqual(np.shape(preds), (3,))
    self.assertTrue(np.all((preds >= 0.0) & (preds <= 1.0)))

  @parameterized.named_parameters(BASE_CLASSIFIERS)
  def test_classifiers_can_predict_if_fitted_multiclass_data(
      self, base_model_cls, model_params
  ):
    model = base_model_cls(**model_params)
    model.fit(self.multiclass_data)
    preds = model.predict(self.pred_data)

    self.assertIsInstance(preds, np.ndarray)
    self.assertEqual(preds.dtype, np.float32)
    self.assertEqual(np.shape(preds), (3, 3))
    self.assertTrue(np.all((preds >= 0.0) & (preds <= 1.0)))
    np.testing.assert_allclose(preds.sum(axis=1), np.ones(3), atol=1e-5)

  @parameterized.named_parameters(BASE_REGRESSORS + BASE_CLASSIFIERS)
  def test_models_cannot_predict_if_not_fitted(
      self, base_model_cls, model_params
  ):
    model = base_model_cls(**model_params)
    with self.assertRaises(RuntimeError):
      model.predict(self.pred_data)

  @parameterized.named_parameters(BASE_CLASSIFIERS + BASE_REGRESSORS)
  def test_models_can_export(self, base_model_cls, model_params):
    model = base_model_cls(**model_params)
    output_model = model.export()
    self.assertIsNotNone(output_model)

  @parameterized.named_parameters(BASE_CLASSIFIERS + BASE_REGRESSORS)
  def test_can_get_params_from_all_models(self, base_model_cls, model_params):
    model = base_model_cls(**model_params)
    params_out = model.get_params()

    default_params = {
        k: v.default
        for k, v in inspect.signature(
            base_model_cls.__init__
        ).parameters.items()
        if v.default is not inspect.Parameter.empty
    }
    all_params = default_params | model_params

    self.assertEqual(params_out, all_params)

  @parameterized.named_parameters(BASE_REGRESSORS)
  def test_regression_model_not_used_if_labels_constant(
      self, base_model_cls, model_params
  ):
    model = base_model_cls(**model_params)
    model._fit = mock.MagicMock()
    model._predict = mock.MagicMock()

    model.fit(self.regression_data_const)
    model._fit.assert_not_called()

    preds = model.predict(self.pred_data)
    model._predict.assert_not_called()

    np.testing.assert_allclose(preds, 0.5 * np.ones(3), atol=1e-5)

  @parameterized.named_parameters(BASE_CLASSIFIERS)
  def test_classification_model_not_used_if_labels_constant(
      self, base_model_cls, model_params
  ):
    model = base_model_cls(**model_params)
    model._fit = mock.MagicMock()
    model._predict = mock.MagicMock()

    model.fit(self.binomial_data_const)
    model._fit.assert_not_called()

    preds = model.predict(self.pred_data)
    model._predict.assert_not_called()

    np.testing.assert_allclose(preds, np.ones(3), atol=1e-5)


if __name__ == "__main__":
  absltest.main()
