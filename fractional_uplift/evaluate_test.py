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

"""Tests for the evaluate module."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import pandas as pd
from unittest import mock

from fractional_uplift import evaluate


EffectType = evaluate.EffectType


class UpliftEvaluatorTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.test_data = pd.DataFrame({
        "metric_1": np.linspace(0.0, 1.0, 12),
        "metric_2": np.linspace(1.0, 0.0, 12),
        "score_1": np.linspace(0.0, 1.0, 12),
        "score_2": np.linspace(1.0, 0.0, 12),
        "is_treated": [0, 1] * 6,
        "treatment_propensity": np.linspace(0.2, 0.8, 12),
    })
    self.default_init_params = dict(
        n_bins=3,
        is_treated_col="is_treated",
        treatment_propensity_col="treatment_propensity",
        metric_cols=["metric_1", "metric_2"],
    )
    self.default_score_cols = ["score_1", "score_2"]

  def test_changing_effect_type_changes_results(self):
    ate_evaluator = evaluate.UpliftEvaluator(
        effect_type=EffectType.ATE, **self.default_init_params
    )
    att_evaluator = evaluate.UpliftEvaluator(
        effect_type=EffectType.ATT, **self.default_init_params
    )
    atc_evaluator = evaluate.UpliftEvaluator(
        effect_type=EffectType.ATC, **self.default_init_params
    )

    ate_results = ate_evaluator.evaluate(
        self.test_data, self.default_score_cols
    )
    att_results = att_evaluator.evaluate(
        self.test_data, self.default_score_cols
    )
    atc_results = atc_evaluator.evaluate(
        self.test_data, self.default_score_cols
    )

    self.assertFalse(np.array_equal(ate_results.values, att_results.values))
    self.assertFalse(np.array_equal(att_results.values, atc_results.values))
    self.assertFalse(np.array_equal(atc_results.values, ate_results.values))

  def test_att_and_atc_equal_if_treatment_propensity_is_05(self):
    self.test_data["treatment_propensity"] = 0.5

    att_evaluator = evaluate.UpliftEvaluator(
        effect_type=EffectType.ATT, **self.default_init_params
    )
    atc_evaluator = evaluate.UpliftEvaluator(
        effect_type=EffectType.ATC, **self.default_init_params
    )

    att_results = att_evaluator.evaluate(
        self.test_data, self.default_score_cols
    )
    atc_results = atc_evaluator.evaluate(
        self.test_data, self.default_score_cols
    )

    pd.testing.assert_frame_equal(att_results, atc_results)

  def test_correct_weights_used_for_ate_effect_type(self):
    evaluator = evaluate.UpliftEvaluator(
        effect_type=EffectType.ATE, **self.default_init_params
    )
    is_treated = self.test_data["is_treated"].values == 1
    treatment_propensity = self.test_data["treatment_propensity"].values

    propensity_weights = evaluator._calculate_inverse_propensity_weights(
        self.test_data
    )

    np.testing.assert_allclose(
        propensity_weights[is_treated],
        1.0 / treatment_propensity[is_treated],
    )
    np.testing.assert_allclose(
        propensity_weights[~is_treated],
        1.0 / (1.0 - treatment_propensity[~is_treated]),
    )

  def test_correct_weights_used_for_att_effect_type(self):
    evaluator = evaluate.UpliftEvaluator(
        effect_type=EffectType.ATT, **self.default_init_params
    )
    is_treated = self.test_data["is_treated"].values == 1
    treatment_propensity = self.test_data["treatment_propensity"].values

    propensity_weights = evaluator._calculate_inverse_propensity_weights(
        self.test_data
    )

    np.testing.assert_allclose(
        propensity_weights[is_treated],
        np.ones_like(treatment_propensity[is_treated]),
    )
    np.testing.assert_allclose(
        propensity_weights[~is_treated],
        treatment_propensity[~is_treated]
        / (1.0 - treatment_propensity[~is_treated]),
    )

  def test_correct_weights_used_for_atc_effect_type(self):
    evaluator = evaluate.UpliftEvaluator(
        effect_type=EffectType.ATC, **self.default_init_params
    )
    is_treated = self.test_data["is_treated"].values == 1
    treatment_propensity = self.test_data["treatment_propensity"].values

    propensity_weights = evaluator._calculate_inverse_propensity_weights(
        self.test_data
    )

    np.testing.assert_allclose(
        propensity_weights[is_treated],
        (1.0 - treatment_propensity[is_treated])
        / treatment_propensity[is_treated],
    )
    np.testing.assert_allclose(
        propensity_weights[~is_treated],
        np.ones_like(treatment_propensity[is_treated]),
    )

  @parameterized.parameters(2, 3, 4, 5)
  def test_n_bins_sets_the_number_of_bins(self, n_bins):
    del self.default_init_params["n_bins"]
    evaluator = evaluate.UpliftEvaluator(
        n_bins=n_bins, **self.default_init_params
    )

    results = evaluator.evaluate(self.test_data, self.default_score_cols)
    bin_counts = results.groupby("name").count()["share_targeted"].values

    np.testing.assert_array_equal(bin_counts, n_bins * np.ones_like(bin_counts))

  @parameterized.parameters(
      {"metric_cols": ["metric_1"]},
      {"metric_cols": ["metric_2"]},
      {"metric_cols": ["metric_1", "metric_2"]},
  )
  def test_effects_are_calculated_for_all_metric_cols(self, metric_cols):
    del self.default_init_params["metric_cols"]
    evaluator = evaluate.UpliftEvaluator(
        metric_cols=metric_cols, **self.default_init_params
    )

    results = evaluator.evaluate(self.test_data, self.default_score_cols)

    expected_col_names = (
        ["name", "share_targeted", "avg_score"]
        + [f"{metric}__inc" for metric in metric_cols]
        + [f"{metric}__inc_cum" for metric in metric_cols]
        + [f"{metric}__control" for metric in metric_cols]
        + [f"{metric}__treated" for metric in metric_cols]
        + [f"{metric}__control_cum" for metric in metric_cols]
        + [f"{metric}__treated_cum" for metric in metric_cols]
    )
    self.assertEqual(set(expected_col_names), set(results.columns.values))

  @parameterized.parameters(
      {"score_cols": ["score_1"]},
      {"score_cols": ["score_2"]},
      {"score_cols": ["score_1", "score_2"]},
  )
  def test_effects_are_calculated_for_all_score_cols(self, score_cols):
    evaluator = evaluate.UpliftEvaluator(**self.default_init_params)

    results = evaluator.evaluate(self.test_data, score_cols)

    expected_model_names = ["random"] + score_cols
    self.assertEqual(set(expected_model_names), set(results["name"].values))

  def test_evaluate_calculates_incrementality_per_bin(self):
    evaluator = evaluate.UpliftEvaluator(**self.default_init_params)

    results = evaluator.evaluate(self.test_data, self.default_score_cols)

    for metric in ["metric_1", "metric_2"]:
      np.testing.assert_allclose(
          results[f"{metric}__inc"].values,
          results[f"{metric}__treated"].values
          - results[f"{metric}__control"].values,
      )

  def test_evaluate_calculates_cumulative_incrementality_per_bin(self):
    evaluator = evaluate.UpliftEvaluator(**self.default_init_params)

    results = evaluator.evaluate(self.test_data, self.default_score_cols)

    for _, results_group in results.groupby("name"):
      results_group = results_group.sort_values("share_targeted")
      for metric in [
          "metric_1__inc",
          "metric_1__control",
          "metric_1__treated",
          "metric_2__inc",
          "metric_2__control",
          "metric_2__treated",
      ]:
        np.testing.assert_allclose(
            results_group[f"{metric}_cum"].values,
            results_group[metric].values.cumsum(),
        )

  def test_composite_metrics_can_be_calculated_by_subclassing(self):
    class EvaluatorWithCompositeMetrics(evaluate.UpliftEvaluator):
      """Example of an evaluator with composite metrics."""

      def _calculate_composite_metrics(
          self, data: pd.DataFrame
      ) -> pd.DataFrame:
        data["my_new_metric"] = (
            data["metric_1__inc_cum"] - data["metric_2__inc_cum"]
        )
        return data

    evaluator = EvaluatorWithCompositeMetrics(**self.default_init_params)
    results = evaluator.evaluate(self.test_data, self.default_score_cols)

    np.testing.assert_allclose(
        results["my_new_metric"].values,
        results["metric_1__inc_cum"].values
        - results["metric_2__inc_cum"].values,
    )


if __name__ == "__main__":
  absltest.main()
