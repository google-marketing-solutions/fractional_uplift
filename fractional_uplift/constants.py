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

"""A collection of constants used throughout the package.

Constants are used in multiple modules throughout the package.
"""
import enum
import tensorflow_decision_forests as tfdf


Task = tfdf.keras.core.Task
TaskType = tfdf.keras.core.TaskType


class ColumnName(enum.Enum):
  """Column names used when constructing dataframes."""

  LABEL = "label_"
  WEIGHT = "weight_"
  MAXIMIZE_KPI = "maximize_kpi_"
  CONSTRAINT_KPI = "constraint_kpi_"
  CONSTRAINT_OFFSET_KPI = "constraint_offset_kpi_"
  IS_TREATED = "is_treated_"
  TREATMENT_PROPENSITY = "treatment_propensity_"


class KPI(enum.Enum):
  """Indexes for each kpi type.

  Used to create the multi-class classification problem.
  """

  MAXIMIZE_KPI = 0
  CONSTRAINT_KPI = 1
  CONSTRAINT_OFFSET_KPI = 2


MAX_N_CLASSES = 3


class EffectType(enum.Enum):
  """Types of effect that can be used for evaluation."""

  ATE = "average_treatment_effect"
  ATT = "average_treatment_effect_on_treated"
  ATC = "average_treatment_effect_on_control"
