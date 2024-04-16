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

from fractional_uplift.data_processor_api import datasets


Dataset = datasets.Dataset


class DataProcessor:
  """A class for processing a dataset.

  This class produces processed copies of the input dataset, which are processed
  correctly for the meta learners to operate.
  """

  def __init__(
      self,
      input_dataset: Dataset,
      *,
      maximize_kpi_column: str,
      is_treated_column: str,
      treatment_propensity_column: str | None = None,
      constraint_kpi_column: str | None = None,
      constraint_offset_kpi_column: str | None = None,
      sample_weight_column: str | None = None,
  ):
    """Initializes the data processor.

    Args:
      input_dataset: The dataset to process.
      maximize_kpi_column: The column name of the KPI to maximize.
      is_treated_column: The column name of the treatment indicator.
      treatment_propensity_column: The column name of the treatment propensity.
      constraint_kpi_column: The column name of the KPI to constrain.
      constraint_offset_kpi_column: The column name of the offset for the
        constraint KPI.
      sample_weight_column: The column name of the sample weight.
    """
    self.input_dataset = input_dataset
    self.maximize_kpi_column = maximize_kpi_column
    self.is_treated_column = is_treated_column
    self.treatment_propensity_column = treatment_propensity_column
    self.constraint_kpi_column = constraint_kpi_column
    self.constraint_offset_kpi_column = constraint_offset_kpi_column
    self.sample_weight_column = sample_weight_column

    self.other_columns = list(
        filter(None, [
            self.maximize_kpi_column,
            self.is_treated_column,
            self.treatment_propensity_column,
            self.constraint_kpi_column,
            self.constraint_offset_kpi_column,
            self.sample_weight_column,
        ])
    )

    self.feature_columns = [
        c
        for c in self.input_dataset.get_columns()
        if c not in self.other_columns
    ]

    self._validate_input_columns_exist()

  def _is_missing(
      self, column_name: str | None, required: bool = False
  ) -> bool:
    """Returns true if the column is missing from the input_dataset.

    If the column name is None and it is not required, then it is not considered
    missing. However if it is None and it is required then it is considered
    missing.

    Args:
      column_name: The name of the column to check.
      required: If true, the column must not be None.

    Returns:
      True if the column is missing from the dataset, false otherwise.
    """
    if column_name is None and not required:
      return False
    return not self.input_dataset.column_exists(column_name)

  def _validate_input_columns_exist(self) -> None:
    """Validates that the input columns exist in the dataset."""

    if self._is_missing(self.maximize_kpi_column, required=True):
      raise ValueError(
          f'maximize_kpi_column {self.maximize_kpi_column} does not exist in'
          ' the dataset.'
      )

    if self._is_missing(self.is_treated_column, required=True):
      raise ValueError(
          f'is_treated_column {self.is_treated_column} does not exist in the'
          ' dataset.'
      )

    if self._is_missing(self.treatment_propensity_column):
      raise ValueError(
          f'treatment_propensity_column {self.treatment_propensity_column} does'
          ' not exist in the dataset.'
      )

    if self._is_missing(self.constraint_kpi_column):
      raise ValueError(
          f'constraint_kpi_column {self.constraint_kpi_column} does not exist'
          ' in the dataset.'
      )

    if self._is_missing(self.constraint_offset_kpi_column):
      raise ValueError(
          'constraint_offset_kpi_column'
          f' {self.constraint_offset_kpi_column} does not exist in the dataset.'
      )

    if self._is_missing(self.sample_weight_column):
      raise ValueError(
          f'sample_weight_column {self.sample_weight_column} does not exist in'
          ' the dataset.'
      )
