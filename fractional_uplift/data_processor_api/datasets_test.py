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


if __name__ == "__main__":
  absltest.main()
