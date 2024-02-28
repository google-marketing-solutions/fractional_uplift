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

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import pandas as pd

from fractional_uplift import example_data


def _no_downsample(
    data: pd.DataFrame, keep_fraction: float = 1.0, seed: int = 1234
) -> pd.DataFrame:
  """Used for mocking to skip downsampling of test data."""
  return data


class TestExampleDataUtils(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    example_data.clear_dataframe_cache()

  def test__csv_to_dataframe_with_cache_reads_csv_into_cache_and_returns_copy(
      self,
  ):
    mock_path = "test/path"
    mock_data = mock.MagicMock(spec=pd.DataFrame)
    mock_data_copy = mock.MagicMock(spec=pd.DataFrame)
    mock_data.copy.return_value = mock_data_copy

    with mock.patch.object(
        example_data.pd, "read_csv", return_value=mock_data
    ) as mock_read_csv:
      output = example_data._csv_to_dataframe_with_cache(mock_path)

    mock_read_csv.assert_called_once_with(mock_path)
    self.assertIs(output, mock_data_copy)
    self.assertIs(example_data._DATAFRAME_CACHE[mock_path], mock_data)

  def test_clear_dataframe_cache_removes_dataframes_from_cache(self):
    example_data._DATAFRAME_CACHE["test/path"] = mock.MagicMock()
    example_data.clear_dataframe_cache()
    self.assertEmpty(example_data._DATAFRAME_CACHE.keys())

  def test_csv_only_loaded_from_url_on_first_call(self):
    mock_path = "test/path"
    mock_data = mock.MagicMock(spec=pd.DataFrame)
    mock_data_copy = mock.MagicMock(spec=pd.DataFrame)
    mock_data.copy.return_value = mock_data_copy

    with mock.patch.object(
        example_data.pd, "read_csv", return_value=mock_data
    ) as mock_read_csv:
      example_data._csv_to_dataframe_with_cache(mock_path)
      example_data._csv_to_dataframe_with_cache(mock_path)

    mock_read_csv.assert_called_once_with(mock_path)

  @parameterized.parameters(0.0, 0.5, 0.8, 1.0)
  def test_downsample_non_converters_downsamples(self, keep_fraction):
    n_converters = 2000
    n_non_converters = 2000
    input_data = pd.DataFrame({
        "conversion": [0] * n_non_converters + [1] * n_converters,
        "other_col": np.random.randn(n_converters + n_non_converters),
    })

    output_data = example_data._downsample_non_converters(
        input_data, keep_fraction=keep_fraction
    )

    n_converters_kept = np.sum(output_data["conversion"])
    n_non_converters_kept = np.sum(1.0 - output_data["conversion"])

    self.assertEqual(n_converters_kept, n_converters)
    if keep_fraction == 0.0:
      self.assertEqual(n_non_converters_kept, 0)
    elif keep_fraction == 1.0:
      self.assertEqual(n_non_converters_kept, n_non_converters)
    else:
      self.assertBetween(
          n_non_converters_kept,
          n_non_converters * keep_fraction * 0.9,
          n_non_converters * keep_fraction * 1.1,
      )

  @parameterized.parameters(0.5, 0.8, 1.0)
  def test_downsample_non_converters_sets_sample_weight_as_inverse_of_keep_frac(
      self, keep_fraction
  ):
    n_converters = 2000
    n_non_converters = 2000
    input_data = pd.DataFrame({
        "conversion": [0] * n_non_converters + [1] * n_converters,
        "other_col": np.random.randn(n_converters + n_non_converters),
    })

    output_data = example_data._downsample_non_converters(
        input_data, keep_fraction=keep_fraction
    )

    is_converted = output_data["conversion"] == 1
    n_converters_kept = int(np.sum(output_data["conversion"]))
    n_non_converters_kept = int(np.sum(1 - output_data["conversion"]))

    np.testing.assert_equal(
        output_data.loc[is_converted, "sample_weight"].values,
        np.ones(n_converters_kept),
    )
    np.testing.assert_equal(
        output_data.loc[~is_converted, "sample_weight"].values,
        np.ones(n_non_converters_kept) / keep_fraction,
    )


class TestCriteoData(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_criteo_data = pd.DataFrame({
        "conversion": [True, True, False, False],
        "exposure": [True, False, True, False],
        "f0": [0.1, 0.2, 0.3, 0.4],
        "f1": [0.4, 0.5, 0.6, 0.7],
        "f10": [0.8, 0.9, 0.1, 1.0],
        "f11": [0.1, 0.2, 0.3, 0.4],
        "f2": [0.4, 0.5, 0.6, 0.7],
        "f3": [0.8, 0.9, 0.1, 1.0],
        "f4": [-0.2, -0.3, -0.4, -0.5],
        "f5": [-1.5, -2.0, -3.0, -4.0],
        "f6": [1.0, 2.0, 5.0, -1.0],
        "f7": [1.0, -2.0, 0.1, 0.2],
        "f8": [1.0, -2.0, 0.4, 0.5],
        "f9": [-0.1, 0.2, 0.3, 0.4],
        "treatment": [1, 0, 1, 0],
        "visit": [True, True, False, True],
    })

  def test_csv_loaded_from_expected_url(self):
    with mock.patch.object(
        example_data,
        "_csv_to_dataframe_with_cache",
        return_value=self.mock_criteo_data,
    ) as mock__csv_to_dataframe_with_cache:
      _ = example_data.CriteoWithSyntheticCostAndSpend.load()

      mock__csv_to_dataframe_with_cache.assert_called_with(
          "http://go.criteo.net/criteo-research-uplift-v2.1.csv.gz"
      )

  def test_non_converters_are_downsampled_in_train_distill_and_test(self):
    with mock.patch.object(
        example_data,
        "_csv_to_dataframe_with_cache",
        return_value=self.mock_criteo_data,
    ):
      criteo = example_data.CriteoWithSyntheticCostAndSpend.load()

    tot_converters_after_downsample = (
        criteo.train_data["conversion"].sum()
        + criteo.distill_data["conversion"].sum()
        + criteo.test_data["conversion"].sum()
    )
    tot_non_converters_after_downsample = (
        (1 - criteo.train_data["conversion"]).sum()
        + (1 - criteo.distill_data["conversion"]).sum()
        + (1 - criteo.test_data["conversion"]).sum()
    )
    self.assertEqual(tot_converters_after_downsample, 2)
    self.assertLess(tot_non_converters_after_downsample, 2)

  def test_features_columns_are_unchanged(self):
    with mock.patch.object(
        example_data,
        "_csv_to_dataframe_with_cache",
        return_value=self.mock_criteo_data,
    ):
      criteo = example_data.CriteoWithSyntheticCostAndSpend.load()

    expected_features = [
        c for c in self.mock_criteo_data.columns if c.startswith("f")
    ]
    pd.testing.assert_frame_equal(
        criteo.data[criteo.features],
        self.mock_criteo_data[expected_features],
        check_like=True,
    )

  @parameterized.parameters("treatment", "conversion")
  def test_treatment_and_conversion_columns_are_cast_to_integer(
      self, column_name
  ):
    with mock.patch.object(
        example_data,
        "_csv_to_dataframe_with_cache",
        return_value=self.mock_criteo_data,
    ):
      criteo = example_data.CriteoWithSyntheticCostAndSpend.load()

    np.testing.assert_equal(
        criteo.data[column_name].values,
        self.mock_criteo_data[column_name].values.astype(np.int64),
    )

  @parameterized.parameters("visit", "exposure")
  def test_visit_and_exposure_columns_are_dropped(self, column_name):
    with mock.patch.object(
        example_data,
        "_csv_to_dataframe_with_cache",
        return_value=self.mock_criteo_data,
    ):
      criteo = example_data.CriteoWithSyntheticCostAndSpend.load()

    self.assertNotIn(column_name, criteo.data.columns)

  def test_treatment_propensity_column_is_added(self):
    with mock.patch.object(
        example_data,
        "_csv_to_dataframe_with_cache",
        return_value=self.mock_criteo_data,
    ):
      criteo = example_data.CriteoWithSyntheticCostAndSpend.load()

    np.testing.assert_equal(
        criteo.data["treatment_propensity"].values,
        np.array([0.5, 0.5, 0.5, 0.5]),
    )

  def test_spend_is_positive_for_converting_users(self):
    with mock.patch.object(
        example_data,
        "_csv_to_dataframe_with_cache",
        return_value=self.mock_criteo_data,
    ):
      criteo = example_data.CriteoWithSyntheticCostAndSpend.load()

    is_converted = criteo.data["conversion"] == 1
    self.assertTrue(np.all(criteo.data.loc[is_converted, "spend"].values > 0))

  def test_spend_is_positive_for_converting_users(self):
    with mock.patch.object(
        example_data,
        "_csv_to_dataframe_with_cache",
        return_value=self.mock_criteo_data,
    ):
      criteo = example_data.CriteoWithSyntheticCostAndSpend.load()

    is_converted = criteo.data["conversion"] == 1
    is_treated = criteo.data["treatment"] == 1
    self.assertTrue(
        np.all(criteo.data.loc[is_converted & is_treated, "cost"].values > 0)
    )

  @parameterized.parameters("cost", "spend")
  def test_cost_and_spend_are_zero_for_non_converting_users(self, column_name):
    with mock.patch.object(
        example_data,
        "_csv_to_dataframe_with_cache",
        return_value=self.mock_criteo_data,
    ):
      criteo = example_data.CriteoWithSyntheticCostAndSpend.load()

    is_converted = criteo.data["conversion"] == 0
    n_converted = np.sum(is_converted)
    np.testing.assert_equal(
        criteo.data.loc[is_converted, column_name].values, np.zeros(n_converted)
    )

  def test_data_is_split_into_train_distill_and_test(self):
    with mock.patch.object(
        example_data,
        "_csv_to_dataframe_with_cache",
        return_value=self.mock_criteo_data,
    ):
      with mock.patch.object(
          example_data, "_downsample_non_converters", side_effect=_no_downsample
      ):
        criteo = example_data.CriteoWithSyntheticCostAndSpend.load()

    pd.testing.assert_frame_equal(
        criteo.data,
        pd.concat([criteo.train_data, criteo.distill_data, criteo.test_data]),
        check_like=True,
    )


if __name__ == "__main__":
  absltest.main()
