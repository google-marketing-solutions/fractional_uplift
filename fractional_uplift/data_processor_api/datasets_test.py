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
import numpy as np
import pandas as pd

from fractional_uplift.data_processor_api import datasets


class PandasDatasetTest(parameterized.TestCase):

  def test_dataset_is_initialised(self):
    input_data = pd.DataFrame({"col_1": [1, 2, 3]})
    data = datasets.PandasDataset(input_data)
    pd.testing.assert_frame_equal(
        data.as_pd_dataframe(),
        input_data,
    )

  def test_column_is_not_negative_returns_true_for_non_negative_column(self):
    input_data = pd.DataFrame({"col_1": [0.0, 1.0, 2.0, 3.0]})
    data = datasets.PandasDataset(input_data)
    self.assertTrue(data.column_is_not_negative("col_1"))

  def test_column_is_not_negative_returns_false_for_negative_column(self):
    input_data = pd.DataFrame({"col_1": [-1.0, 2.0, 3.0]})
    data = datasets.PandasDataset(input_data)
    self.assertFalse(data.column_is_not_negative("col_1"))

  def test_column_is_not_negative_returns_false_for_nan_values(self):
    input_data = pd.DataFrame({"col_1": [np.nan, 2.0, 3.0]})
    data = datasets.PandasDataset(input_data)
    self.assertFalse(data.column_is_not_negative("col_1"))

  def test_column_exists_returns_true_if_column_exists(self):
    input_data = pd.DataFrame({"col_1": [1.0, 2.0, 3.0]})
    data = datasets.PandasDataset(input_data)
    self.assertTrue(data.column_exists("col_1"))

  def test_column_exists_returns_false_if_column_not_exists(self):
    input_data = pd.DataFrame({"col_1": [1.0, 2.0, 3.0]})
    data = datasets.PandasDataset(input_data)
    self.assertFalse(data.column_exists("col_2"))

  def test_copy_makes_a_copy_of_the_data(self):
    input_data = pd.DataFrame({"col_1": [1.0, 2.0, 3.0]})
    data = datasets.PandasDataset(input_data)
    data_copy = data.copy()

    pd.testing.assert_frame_equal(
        data.as_pd_dataframe(), data_copy.as_pd_dataframe()
    )
    self.assertIsNot(data_copy, data)
    self.assertIsNot(data.as_pd_dataframe(), data_copy.as_pd_dataframe())

  def test_set_column_from_constant_sets_the_column_correctly(self):
    input_data = pd.DataFrame({"col_1": [1.0, 2.0, 3.0]})
    data = datasets.PandasDataset(input_data)

    data_output = data.set_column_from_constant("col_2", 5)

    expected_output_data = pd.DataFrame(
        {"col_1": [1.0, 2.0, 3.0], "col_2": [5, 5, 5]}
    )
    pd.testing.assert_frame_equal(
        data_output.as_pd_dataframe(), expected_output_data
    )

  def test_set_column_from_numpy_array_sets_the_column_correctly(self):
    input_data = pd.DataFrame({"col_1": [1.0, 2.0, 3.0]})
    data = datasets.PandasDataset(input_data)

    data_output = data.set_column_from_constant("col_2", np.array([5, 6, 7]))

    expected_output_data = pd.DataFrame(
        {"col_1": [1.0, 2.0, 3.0], "col_2": [5, 6, 7]}
    )
    pd.testing.assert_frame_equal(
        data_output.as_pd_dataframe(), expected_output_data
    )

  def test_filter_filters_for_true_elements_of_the_mask_column(self):
    input_data = pd.DataFrame(
        {"col_1": [1.0, 2.0, 3.0], "col_2": [True, True, False]}
    )
    data = datasets.PandasDataset(input_data)

    data_output = data.filter("col_2")

    expected_output_data = pd.DataFrame(
        {"col_1": [1.0, 2.0], "col_2": [True, True]}
    )
    pd.testing.assert_frame_equal(
        data_output.as_pd_dataframe(), expected_output_data
    )

  def test_get_columns_returns_a_list_of_column_names(self):
    input_data = pd.DataFrame(
        {"col_1": [1.0, 2.0, 3.0], "col_2": [True, True, False]}
    )
    data = datasets.PandasDataset(input_data)

    self.assertListEqual(data.get_columns(), ["col_1", "col_2"])

  def test_append_rows_appends_the_second_dataframe_to_the_first(self):
    input_data_1 = pd.DataFrame(
        {"col_1": [1.0, 2.0, 3.0], "col_2": ["a", "b", "c"]}
    )
    data_1 = datasets.PandasDataset(input_data_1)

    input_data_2 = pd.DataFrame(
        {"col_1": [8.0, 9.0, 10.0], "col_2": ["g", "h", "i"]}
    )
    data_2 = datasets.PandasDataset(input_data_2)

    output_data = data_1.append_rows(data_2)

    expected_output_data = pd.DataFrame({
        "col_1": [1.0, 2.0, 3.0, 8.0, 9.0, 10.0],
        "col_2": ["a", "b", "c", "g", "h", "i"],
    })
    pd.testing.assert_frame_equal(
        output_data.as_pd_dataframe(), expected_output_data
    )

  def test_append_rows_aligns_the_columns_of_the_second_dataset_with_the_first(
      self,
  ):
    input_data_1 = pd.DataFrame(
        {"col_1": [1.0, 2.0, 3.0], "col_2": ["a", "b", "c"]}
    )
    data_1 = datasets.PandasDataset(input_data_1)

    input_data_2 = pd.DataFrame(
        {"col_2": ["g", "h", "i"], "col_1": [8.0, 9.0, 10.0]}
    )
    data_2 = datasets.PandasDataset(input_data_2)

    output_data = data_1.append_rows(data_2)

    expected_output_data = pd.DataFrame({
        "col_1": [1.0, 2.0, 3.0, 8.0, 9.0, 10.0],
        "col_2": ["a", "b", "c", "g", "h", "i"],
    })
    pd.testing.assert_frame_equal(
        output_data.as_pd_dataframe(), expected_output_data
    )

  def test_append_rows_raise_excption_if_columns_dont_match(self):
    input_data_1 = pd.DataFrame(
        {"col_1": [1.0, 2.0, 3.0], "col_2": ["a", "b", "c"]}
    )
    data_1 = datasets.PandasDataset(input_data_1)

    input_data_2 = pd.DataFrame({
        "col_1": [8.0, 9.0, 10.0],
        "col_2": ["g", "h", "i"],
        "col_3": [1.0, 2.0, 3.0],
    })
    data_2 = datasets.PandasDataset(input_data_2)

    with self.assertRaisesRegex(
        ValueError, "The columns don't match between the two datasets:"
    ):
      data_1.append_rows(data_2)


if __name__ == "__main__":
  absltest.main()
