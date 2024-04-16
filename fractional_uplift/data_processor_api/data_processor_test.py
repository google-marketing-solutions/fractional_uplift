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

from absl.testing import absltest
from absl.testing import parameterized
import pandas as pd

from fractional_uplift.data_processor_api import data_processor
from fractional_uplift.data_processor_api import datasets


class DataProcessorTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    self.default_input_data = pd.DataFrame({
        "X1": [1.0, 2.0, 3.0, 4.0],
        "X2": [6, 7, 8, 9],
        "X3": ["a", "b", "c", "d"],
        "my_maximize_kpi": [1.0, 5.0, 3.0, -1.0],
        "my_constraint_kpi": [4.0, 2.0, 1.5, 6.0],
        "my_constraint_offset_kpi": [2.0, 1.0, 1.5, 2.0],
        "treatment_propensity": [0.6, 0.5, 0.3, 0.6],
        "is_treated": [1, 1, 0, 0],
        "sample_weight": [1.0, 2.0, 2.0, 1.0],
    })

  def test_init_sets_other_columns_as_all_specified_columns(self):
    dataset = datasets.PandasDataset(self.default_input_data)
    processor = data_processor.DataProcessor(
        dataset,
        maximize_kpi_column="my_maximize_kpi",
        constraint_kpi_column="my_constraint_kpi",
        constraint_offset_kpi_column="my_constraint_offset_kpi",
        treatment_propensity_column="treatment_propensity",
        is_treated_column="is_treated",
        sample_weight_column="sample_weight",
    )

    self.assertCountEqual(
        processor.other_columns,
        [
            "my_maximize_kpi",
            "my_constraint_kpi",
            "my_constraint_offset_kpi",
            "treatment_propensity",
            "is_treated",
            "sample_weight",
        ],
    )

  def test_init_ignores_none_columns_in_other_columns(self):
    dataset = datasets.PandasDataset(self.default_input_data)
    processor = data_processor.DataProcessor(
        dataset,
        maximize_kpi_column="my_maximize_kpi",
        is_treated_column="is_treated",
    )

    self.assertCountEqual(
        processor.other_columns, ["my_maximize_kpi", "is_treated"]
    )

  def test_init_sets_all_unspecified_columns_as_feature_columns(self):
    dataset = datasets.PandasDataset(self.default_input_data)
    processor = data_processor.DataProcessor(
        dataset,
        maximize_kpi_column="my_maximize_kpi",
        constraint_kpi_column="my_constraint_kpi",
        constraint_offset_kpi_column="my_constraint_offset_kpi",
        treatment_propensity_column="treatment_propensity",
        is_treated_column="is_treated",
        sample_weight_column="sample_weight",
    )

    self.assertCountEqual(processor.feature_columns, ["X1", "X2", "X3"])

  @parameterized.parameters(
      "my_maximize_kpi",
      "my_constraint_kpi",
      "my_constraint_offset_kpi",
      "treatment_propensity",
      "is_treated",
      "sample_weight",
  )
  def test_raises_error_when_columns_dont_exist_in_input_dataset(
      self, missing_column
  ):
    dataset = datasets.PandasDataset(
        self.default_input_data.drop(columns=[missing_column])
    )
    with self.assertRaisesRegex(
        ValueError, f"{missing_column} does not exist in the dataset."
    ):
      data_processor.DataProcessor(
          dataset,
          maximize_kpi_column="my_maximize_kpi",
          constraint_kpi_column="my_constraint_kpi",
          constraint_offset_kpi_column="my_constraint_offset_kpi",
          treatment_propensity_column="treatment_propensity",
          is_treated_column="is_treated",
          sample_weight_column="sample_weight",
      )


if __name__ == "__main__":
  absltest.main()
