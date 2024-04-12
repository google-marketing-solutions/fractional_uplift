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


if __name__ == "__main__":
  absltest.main()