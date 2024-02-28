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

"""Tests that the package can be used end to end."""
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import pandas as pd

from fractional_uplift import fractional_uplift as fr


KPI = fr.constants.KPI


FRACTIONAL_UPLIFT_LEARNERS = [
    dict(
        testcase_name="FractionalRetrospectiveLearner",
        learner=lambda: fr.meta_learners.FractionalRetrospectiveLearner(
            fr.base_models.TensorflowDecisionForestClassifier(
                init_args=dict(num_trees=1, max_depth=1)
            )
        ),
    ),
    dict(
        testcase_name="FractionalTLearner",
        learner=lambda: fr.meta_learners.FractionalLearner(
            fr.base_models.TensorflowDecisionForestRegressor(
                init_args=dict(num_trees=1, max_depth=1)
            )
        ),
    ),
]


FRACTIONAL_UPLIFT_DATA_ARGS = [
    dict(
        testcase_name="no_constraint_offset_kpi",
        data_args=dict(
            maximize_kpi=np.array([0.1, 0.3, 0.6]),
            constraint_kpi=np.array([0.5, 0.2, 0.1]),
            is_treated=np.array([0, 1, 0]),
            treatment_propensity=np.array([0.2, 0.5, 0.1]),
        ),
    ),
    dict(
        testcase_name="constraint_kpi_zero_in_control_no_constraint_offset_kpi",
        data_args=dict(
            maximize_kpi=np.array([0.1, 0.3, 0.6]),
            constraint_kpi=np.array([0.0, 0.2, 0.0]),
            is_treated=np.array([0, 1, 0]),
            treatment_propensity=np.array([0.2, 0.5, 0.1]),
        ),
    ),
    dict(
        testcase_name="with_constraint_offset_kpi",
        data_args=dict(
            maximize_kpi=np.array([0.1, 0.3, 0.6]),
            constraint_kpi=np.array([0.5, 0.2, 0.1]),
            constraint_offset_kpi=np.array([0.0, 3.0, 10.0]),
            is_treated=np.array([0, 1, 0]),
            treatment_propensity=np.array([0.2, 0.5, 0.1]),
        ),
    ),
    dict(
        testcase_name=(
            "constraint_kpi_zero_in_control_with_constraint_offset_kpi"
        ),
        data_args=dict(
            maximize_kpi=np.array([0.1, 0.3, 0.6]),
            constraint_kpi=np.array([0.0, 0.2, 0.0]),
            constraint_offset_kpi=np.array([0.0, 2.0, 10.0]),
            is_treated=np.array([0, 1, 0]),
            treatment_propensity=np.array([0.2, 0.5, 0.1]),
        ),
    ),
    dict(
        testcase_name="with_constraint_offset_kpi_zero_in_control",
        data_args=dict(
            maximize_kpi=np.array([0.1, 0.3, 0.6]),
            constraint_kpi=np.array([0.5, 0.2, 0.1]),
            constraint_offset_kpi=np.array([0.0, 3.0, 0.0]),
            is_treated=np.array([0, 1, 0]),
            treatment_propensity=np.array([0.2, 0.5, 0.1]),
        ),
    ),
    dict(
        testcase_name="constraint_offset_and_constraint_kpi_zero_in_control",
        data_args=dict(
            maximize_kpi=np.array([0.1, 0.3, 0.6]),
            constraint_kpi=np.array([0.0, 0.2, 0.0]),
            constraint_offset_kpi=np.array([0.0, 3.0, 0.0]),
            is_treated=np.array([0, 1, 0]),
            treatment_propensity=np.array([0.2, 0.5, 0.1]),
        ),
    ),
]


FRACTIONAL_UPLIFT_GOOD_CASES = [
    {
        "testcase_name": data_arg["testcase_name"] + learner["testcase_name"],
        "data_args": data_arg["data_args"],
        "learner": learner["learner"],
    }
    for data_arg in FRACTIONAL_UPLIFT_DATA_ARGS
    for learner in FRACTIONAL_UPLIFT_LEARNERS
]


SINGLE_KPI_LEARNERS = [
    dict(
        testcase_name="RetrospectiveLearner",
        learner=lambda: fr.meta_learners.RetrospectiveLearner(
            fr.base_models.TensorflowDecisionForestClassifier(
                init_args=dict(num_trees=1, max_depth=1)
            )
        ),
    ),
    dict(
        testcase_name="TLearner",
        learner=lambda: fr.meta_learners.TLearner(
            fr.base_models.TensorflowDecisionForestRegressor(
                init_args=dict(num_trees=1, max_depth=1)
            )
        ),
    ),
]

SINGLE_KPI_GOOD_CASES = [
    dict(
        testcase_name=f"{learner['testcase_name']}_{kpi.name}",
        kpi=kpi,
        learner=learner["learner"],
    )
    for learner in SINGLE_KPI_LEARNERS
    for kpi in KPI
]


class ImportTest(absltest.TestCase):
  """Tests that the user can import everything."""

  def test_imports(self):
    self.assertIsNotNone(fr.meta_learners)
    self.assertIsNotNone(fr.meta_learners.TLearner)
    self.assertIsNotNone(fr.meta_learners.RetrospectiveLearner)
    self.assertIsNotNone(fr.meta_learners.FractionalLearner)
    self.assertIsNotNone(fr.meta_learners.FractionalRetrospectiveLearner)
    self.assertIsNotNone(fr.datasets)
    self.assertIsNotNone(fr.datasets.PandasTrainData)
    self.assertIsNotNone(fr.datasets.PandasDataset)
    self.assertIsNotNone(fr.base_models)
    self.assertIsNotNone(fr.base_models.TensorflowDecisionForestRegressor)
    self.assertIsNotNone(fr.base_models.TensorflowDecisionForestClassifier)
    self.assertIsNotNone(fr.KPI)
    self.assertIsNotNone(fr.EffectType)
    self.assertIsNotNone(fr.evaluate)
    self.assertIsNotNone(fr.evaluate.UpliftEvaluator)
    self.assertIsNotNone(fr.evaluate.calculate_auc)
    self.assertIsNotNone(fr.example_data)


class MetaLearnersEndToEndlTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    self.pd_dataframe = pd.DataFrame({
        "X1": [1.0, 2.0, 3.0],
        "X2": [1.0, 4.0, 3.0],
        "X3": pd.Categorical(["a", "b", "c"]),
    })

  @parameterized.named_parameters(FRACTIONAL_UPLIFT_GOOD_CASES)
  def test_fractional_learners_can_fit_and_predict(self, data_args, learner):
    train_data = fr.datasets.PandasTrainData(
        features_data=self.pd_dataframe.copy(), shuffle_seed=1234, **data_args
    )
    pred_data = fr.datasets.PandasDataset(
        features_data=self.pd_dataframe.copy(),
    )

    if "constraint_offset_kpi" in data_args.keys():
      constraint_offset_scale = 1.0
    else:
      constraint_offset_scale = None

    model = learner()

    model.fit(train_data)
    preds = model.predict(
        pred_data, constraint_offset_scale=constraint_offset_scale
    )

    self.assertIsInstance(preds, np.ndarray)
    self.assertEqual(preds.dtype, np.float32)
    self.assertEqual(np.shape(preds), (3,))

  @parameterized.named_parameters(SINGLE_KPI_GOOD_CASES)
  def test_single_kpi_learners_can_fit_and_predict(self, kpi, learner):
    data_args = dict(
        features_data=self.pd_dataframe.copy(),
        shuffle_seed=1234,
        is_treated=np.array([0, 1, 0]),
        treatment_propensity=np.array([0.2, 0.5, 0.1]),
        maximize_kpi=np.array([0.1, 0.3, 0.6]),
    )
    if kpi == KPI.CONSTRAINT_KPI:
      data_args["constraint_kpi"] = np.array([0.1, 0.3, 0.6])
    if kpi == KPI.CONSTRAINT_OFFSET_KPI:
      data_args["constraint_kpi"] = np.array([0.1, 0.3, 0.6])
      data_args["constraint_offset_kpi"] = np.array([0.1, 0.3, 0.6])

    train_data = fr.datasets.PandasTrainData(**data_args)
    pred_data = fr.datasets.PandasDataset(
        features_data=self.pd_dataframe.copy(),
    )

    model = learner()
    model.fit(train_data)
    preds = model.predict(pred_data)

    self.assertIsInstance(preds, np.ndarray)
    self.assertEqual(preds.dtype, np.float32)
    self.assertEqual(np.shape(preds), (3,))

  @parameterized.named_parameters(FRACTIONAL_UPLIFT_GOOD_CASES)
  def test_fractional_learners_can_distill(self, data_args, learner):
    train_data = fr.datasets.PandasTrainData(
        features_data=self.pd_dataframe.copy(), shuffle_seed=1234, **data_args
    )
    pred_data = fr.datasets.PandasDataset(
        features_data=self.pd_dataframe.copy(),
    )
    if "constraint_offset_kpi" in data_args.keys():
      constraint_offset_scale = 1.0
    else:
      constraint_offset_scale = None
    model = learner()
    model.fit(train_data)
    distill_model = fr.base_models.TensorflowDecisionForestRegressor()

    model.distill(
        pred_data,
        distill_model,
        constraint_offset_scale=constraint_offset_scale,
    )
    preds = distill_model.predict(pred_data)

    self.assertIsInstance(preds, np.ndarray)
    self.assertEqual(preds.dtype, np.float32)
    self.assertEqual(np.shape(preds), (3,))

  @parameterized.named_parameters(SINGLE_KPI_GOOD_CASES)
  def test_single_kpi_learners_can_distill(self, kpi, learner):
    data_args = dict(
        features_data=self.pd_dataframe.copy(),
        shuffle_seed=1234,
        is_treated=np.array([0, 1, 0]),
        treatment_propensity=np.array([0.2, 0.5, 0.1]),
        maximize_kpi=np.array([0.1, 0.3, 0.6]),
    )
    if kpi == KPI.CONSTRAINT_KPI:
      data_args["constraint_kpi"] = np.array([0.1, 0.3, 0.6])
    if kpi == KPI.CONSTRAINT_OFFSET_KPI:
      data_args["constraint_kpi"] = np.array([0.1, 0.3, 0.6])
      data_args["constraint_offset_kpi"] = np.array([0.1, 0.3, 0.6])

    train_data = fr.datasets.PandasTrainData(**data_args)
    pred_data = fr.datasets.PandasDataset(
        features_data=self.pd_dataframe.copy(),
    )
    model = learner()
    model.fit(train_data)
    distill_model = fr.base_models.TensorflowDecisionForestRegressor()

    model.distill(pred_data, distill_model)
    preds = distill_model.predict(pred_data)

    self.assertIsInstance(preds, np.ndarray)
    self.assertEqual(preds.dtype, np.float32)
    self.assertEqual(np.shape(preds), (3,))


if __name__ == "__main__":
  absltest.main()
