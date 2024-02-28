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

"""Test the base models based on tensorflow decision forests."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import pandas as pd
import tensorflow_decision_forests as tfdf

from fractional_uplift import datasets
from fractional_uplift.base_models import tf_decision_forests


MODELS = [
    dict(
        testcase_name="TensorflowDecisionForestClassifier",
        model_cls=tf_decision_forests.TensorflowDecisionForestClassifier,
        task=tfdf.keras.Task.CLASSIFICATION
    ),
    dict(
        testcase_name="TensorflowDecisionForestRegressor",
        model_cls=tf_decision_forests.TensorflowDecisionForestRegressor,
        task=tfdf.keras.Task.REGRESSION
    ),
]


class TensorflowDecisionForestClassifierTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    self.pd_dataframe = pd.DataFrame({
        "X1": [1.0, 2.0, 3.0],
        "X2": [1.0, 4.0, 3.0],
        "X3": pd.Categorical(["a", "b", "c"]),
    })
    self.dataset_args = {
        "features_data": self.pd_dataframe.copy(),
        "labels": np.array([0, 1, 1]),
        "weights": np.array([2.0, 1.0, 1.0]),
        "shuffle": False,
    }
    self.pred_data = datasets.PandasDataset(
        self.pd_dataframe.copy(), shuffle=False
    )

  @parameterized.named_parameters(MODELS)
  def test_uses_gbm_by_default(self, model_cls, task):
    model = model_cls()
    self.assertIsInstance(
        model._decision_forest, tfdf.keras.GradientBoostedTreesModel
    )

  @parameterized.named_parameters(MODELS)
  def test_user_can_set_different_decision_forest(self, model_cls, task):
    model = model_cls(
        decision_forest_cls=tfdf.keras.RandomForestModel
    )
    self.assertIsInstance(model._decision_forest, tfdf.keras.RandomForestModel)

  @parameterized.named_parameters(MODELS)
  def test_task_is_correct(self, model_cls, task):
    model = model_cls()
    self.assertEqual(
        model._decision_forest.task, task
    )

  @parameterized.named_parameters(MODELS)
  def test_raises_exception_if_task_is_changed(self, model_cls, task):
    with self.assertRaises(ValueError):
      model_cls(
          init_args=dict(task="anything")
      )

  @parameterized.named_parameters(MODELS)
  def test_init_args_are_used(self, model_cls, task):
    with mock.patch.object(
        tf_decision_forests.tfdf.keras,
        "GradientBoostedTreesModel",
        autospec=True,
    ) as mock_gbtm:
      model_cls(
          init_args={"focal_loss_alpha": 0.01}
      )

      mock_gbtm.assert_called_once_with(
          task=task, focal_loss_alpha=0.01, **model_cls.DEFAULT_INIT_ARGS
      )

  @parameterized.named_parameters(MODELS)
  def test_decision_forest_fit_is_called_inside_fit(self, model_cls, task):
    model = model_cls()

    test_data = datasets.PandasDataset(
        task=task,
        **self.dataset_args
    )
    mock_tf_dataset = test_data.as_tf_dataset()

    with mock.patch.object(
        model._decision_forest,
        "fit",
        autospec=True,
    ) as mock_fit:
      with mock.patch.object(
          test_data, "as_tf_dataset", return_value=mock_tf_dataset
      ):
        model.fit(test_data)

        mock_fit.assert_called_once_with(
            mock_tf_dataset, **model_cls.DEFAULT_FIT_ARGS
        )

  @parameterized.named_parameters(MODELS)
  def test_fit_args_are_used(self, model_cls, task):
    fit_args = {"callbacks": "test"}
    model = model_cls(fit_args=fit_args)

    test_data = datasets.PandasDataset(
        task=task,
        **self.dataset_args
    )
    mock_tf_dataset = test_data.as_tf_dataset()

    with mock.patch.object(
        model._decision_forest,
        "fit",
        autospec=True,
    ) as mock_fit:
      with mock.patch.object(
          test_data, "as_tf_dataset", return_value=mock_tf_dataset
      ):
        model.fit(test_data)

        expected_args = model_cls.DEFAULT_FIT_ARGS.copy()
        expected_args.update(fit_args)
        mock_fit.assert_called_once_with(mock_tf_dataset, **expected_args)

  @parameterized.named_parameters(MODELS)
  def test_decision_forest_predict_is_called_inside_predict(
      self, model_cls, task
  ):
    model = model_cls()
    model.fitted = True

    test_data = datasets.PandasDataset(
        task=task,
        **self.dataset_args
    )
    mock_tf_dataset = test_data.as_tf_dataset()
    mock_preds = np.array([0, 1, 2, 3])

    with mock.patch.object(
        model._decision_forest,
        "predict",
        autospec=True,
        return_value=mock_preds,
    ) as mock_predict:
      with mock.patch.object(
          test_data, "as_tf_dataset", return_value=mock_tf_dataset
      ):
        preds = model.predict(test_data)

        mock_predict.assert_called_once_with(mock_tf_dataset)
        np.testing.assert_array_equal(preds, mock_preds)

  @parameterized.named_parameters(MODELS)
  def test_export_returns_decision_forest(self, model_cls, task):
    model = model_cls()
    output_model = model.export()

    self.assertIs(output_model, model._decision_forest)


if __name__ == "__main__":
  absltest.main()
