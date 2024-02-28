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

"""A module containing the classes for creating uplift modelling datasets.

All the datasets to be used for uplift modelling must inherit from this.
It works like an API to allow different data sources to be used to train uplift
models.
"""
from fractional_uplift.datasets import _base
from fractional_uplift.datasets import pandas


Dataset = _base.Dataset
cast_pandas_dataframe_cols = pandas.cast_pandas_dataframe_cols
PandasDataset = pandas.PandasDataset
PandasTrainData = pandas.PandasTrainData
