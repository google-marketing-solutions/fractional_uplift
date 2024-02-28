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

"""Functions for loading data for uplift modeling examples."""

import dataclasses
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn import preprocessing

from fractional_uplift import datasets


_DATAFRAME_CACHE: Dict[str, pd.DataFrame] = dict()
_CRITEO_PATH = "http://go.criteo.net/criteo-research-uplift-v2.1.csv.gz"

_CONVERSION_COL = "conversion"
_TREATMENT_COL = "treatment"
_TREATMENT_PROPENSITY_COL = "treatment_propensity"
_SPEND_COL = "spend"
_COST_COL = "cost"
_COST_PERCENTAGE_COL = "cost_percentage"
_SAMPLE_WEIGHT_COL = "sample_weight"


def _csv_to_dataframe_with_cache(path: str) -> pd.DataFrame:
  """Loads a csv file to pandas dataframe via a cache.

  Useful if the data is being downloaded and is large, so it will only download
  it if it has not already been downloaded.

  Args:
    path: The path to download the csv.

  Returns:
    Data as a pandas dataframe
  """
  if path not in _DATAFRAME_CACHE:
    _DATAFRAME_CACHE[path] = pd.read_csv(path)

  return _DATAFRAME_CACHE[path].copy()


def clear_dataframe_cache() -> None:
  global _DATAFRAME_CACHE
  _DATAFRAME_CACHE = dict()


def _set_treatment_and_conversion_cols(
    data: pd.DataFrame,
    input_treatment_col: str,
    input_conversion_col: str,
) -> pd.DataFrame:
  """Sets the treatment and conversion columns.

  This ensures the treatment and conversion columns are the correct types, and
  creates a treatment_propensity column containing the likelihood for someone
  to be treated.

  Args:
    data: The data to be transformed.
    input_treatment_col: The column indicating if the sample was treated.
    input_conversion_col: The column indicating if the sample converted.

  Returns:
    The transformed data.
  """

  data[_TREATMENT_COL] = data[input_treatment_col].astype(int)
  data[_CONVERSION_COL] = data[input_conversion_col].astype(int)
  data[_TREATMENT_PROPENSITY_COL] = data[_TREATMENT_COL].mean()
  return data


def _create_spend_and_cost_cols(
    data: pd.DataFrame,
    features: List[str],
    seed: int = 47362,
) -> pd.DataFrame:
  """Generates the spend and cost columns using the features in the data.

  For all rows that did not convert, spend and cost are 0. For all rows that did
  convert, the spend and cost are generates as follows:
    1. Normalise all features
    2. Construct polynomial features from input features to introduce
    interactions and non-linearity.
    3. Randomly sample regression weights for the percentage_cost, spend and
    incremental spend, and take the dot product of the polynomial features with
    the weights.
    4. Spend = spend + is_treated * incremental_spend
    5. Cost = percentage_cost * spend

  Args:
    data: The data to add the spend and cost columns.
    features: The list of features which the spend and cost will depend on.
    seed: The random seed to use for sampling.

  Returns:
    The data with the spend and cost columns.
  """
  mask_converters = data[_CONVERSION_COL] == 1
  x = preprocessing.StandardScaler().fit_transform(
      data.loc[mask_converters, features].values
  )
  x_poly = preprocessing.PolynomialFeatures().fit_transform(x)

  np.random.seed(seed)
  w_spend = np.random.randn(np.shape(x_poly)[1])
  y_spend = x_poly @ w_spend.T
  y_spend += np.random.randn(len(y_spend))
  y_spend += 30.0

  w_inc_spend = np.random.randn(np.shape(x_poly)[1])
  y_inc_spend = x_poly @ w_inc_spend.T
  y_inc_spend *= 1.0

  w_cost = np.random.randn(np.shape(x_poly)[1])
  y_cost = x_poly @ w_cost.T

  data[_COST_PERCENTAGE_COL] = 0.0
  data.loc[mask_converters, _COST_PERCENTAGE_COL] = (
      data.loc[mask_converters, _TREATMENT_COL]
      * 1.0
      / (1.0 + np.exp(-0.5 * y_cost + 0.5))
  )

  data[_SPEND_COL] = 0.0
  data.loc[mask_converters, _SPEND_COL] = (
      y_spend + data.loc[mask_converters, _TREATMENT_COL] * y_inc_spend
  )
  spend_lt_5 = data.loc[mask_converters, _SPEND_COL] < 5.0
  data.loc[mask_converters & spend_lt_5, _SPEND_COL] = np.log(
      1.0 + np.exp(data.loc[mask_converters & spend_lt_5, _SPEND_COL])
  )
  data.loc[mask_converters, _SPEND_COL] += 10.0

  data[_COST_COL] = data[_COST_PERCENTAGE_COL] * data[_SPEND_COL]

  return data


def _split_data(
    data: pd.DataFrame,
    seed: int = 47362,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  """Splits the data into train, distill, and test.

  Train is used to train the uplift models.
  Distill is used for fractional uplift models to distill into a single ML
  model.
  Test is used to evaluate the models.

  Because the data is so large, we use 50% of the data for testing, 10% for
  distilling and 40% for training.
  This is probably too much data for testing and too little for training, but it
  makes sure the models don't take too long to train for this example notebook.

  Args:
    data: Input dataframe to be split.
    seed: The random seed to use for sampling.

  Returns:
    train_data, distill_data and test_data.
  """

  np.random.seed(seed)

  is_train = np.random.rand(len(data.index.values)) < 0.5
  is_distill = np.random.rand(len(data.index.values)) < 0.2
  train_data = data.loc[is_train & ~is_distill].copy()
  distill_data = data.loc[is_train & is_distill].copy()
  test_data = data.loc[~is_train].copy()
  return train_data, distill_data, test_data


def _downsample_non_converters(
    data: pd.DataFrame,
    keep_fraction: float,
    seed: int = 47362,
) -> pd.DataFrame:
  """Downsamples the non-converting events.

  Because this is real conversion data, the vast majority of the events are
  non-converting events. Because this function is to generate example data for
  demonstrating the algorithm we want it to run quickly - perfect performance
  isn't the main concern. Therefore, to reduce the data size, we downsample
  the non-converters, and apply a weight to the remaining rows to upweigh them,
  so that the average conversion rate estimated by the models should be
  unchanged.

  Args:
    data: The data to be downsampled.
    keep_fraction: The fraction of the non-converters to keep.
    seed: The random seed to use for sampling.

  Returns:
    The downsampled data.
  """

  np.random.seed(seed)

  downsample = np.random.rand(len(data.index.values)) < keep_fraction
  downsample[data[_CONVERSION_COL] == 1] = True

  # Use sample weights to upweight the non-converters to remove bias
  data[_SAMPLE_WEIGHT_COL] = 1.0
  if keep_fraction > 0.0:
    data.loc[data[_CONVERSION_COL] == 0, _SAMPLE_WEIGHT_COL] = (
        1.0 / keep_fraction
    )

  return data.loc[downsample]


@dataclasses.dataclass(frozen=True)
class CriteoWithSyntheticCostAndSpend:
  """Criteo dataset with synthetic cost and spend columns.

  The criteo dataset is a real dataset for evaluating uplift models. See
  this website for more info:

  https://ailab.criteo.com/criteo-uplift-prediction-dataset/

  We add on top of this two synthetic columns:
  - cost
  - spend

  This is because the original dataset only has columns on whether a customer
  converted, but not how much they spent, or the cost of the treatment, which
  are required for fractional uplift models.

  Attributes:
    features: The list of feature names.
    spend_col: The name of the spend column.
    cost_col: The name of the cost column.
    conversion_col: The name of the conversion column.
    treatment_col: The name of the treatment column.
    data: The full criteo dataset.
    train_data: The criteo dataset to be used for training.
    distill_data: The criteo dataset to be used for distilling.
    test_data: The criteo dataset to be used for testing.
  """

  features: List[str]
  spend_col: str
  cost_col: str
  conversion_col: str
  treatment_col: str

  data: pd.DataFrame
  train_data: pd.DataFrame
  distill_data: pd.DataFrame
  test_data: pd.DataFrame

  @classmethod
  def load(cls, seed: int = 47362) -> "CriteoWithSyntheticCostAndSpend":
    """Loads the Criteo data.

    This performs the following steps:
    1. Download the data.
    2. Transform the boolean columns to integers and add a treatment propensity
    column.
    3. Cast the feature columns to the correct data type.
    4. Create the synthetic spend and cost columns.
    5. Split the data into train, distill, and test.
    6. Downsample the non-converting events (keep 2% for training and
    distilling, and remove all of them for testing).

    Args:
      seed: The random seed to use for sampling.

    Returns:
      The loaded data as an instance of the Criteo class.
    """
    features = [f"f{i}" for i in range(12)]

    data = (
        _csv_to_dataframe_with_cache(_CRITEO_PATH)
        .pipe(
            _set_treatment_and_conversion_cols,
            input_treatment_col="treatment",
            input_conversion_col="conversion",
        )
        .drop(columns=["visit", "exposure"])  # Columns not used
        .pipe(
            datasets.cast_pandas_dataframe_cols,
            num_feature_cols=features,
            cat_feature_cols=[],
        )
        .pipe(_create_spend_and_cost_cols, features, seed=seed)
    )

    train_data, distill_data, test_data = _split_data(data, seed=seed)
    train_data = _downsample_non_converters(
        train_data, keep_fraction=0.01, seed=seed
    )
    distill_data = _downsample_non_converters(
        distill_data, keep_fraction=0.01, seed=seed
    )
    test_data = _downsample_non_converters(
        test_data, keep_fraction=0.0, seed=seed
    )

    return cls(
        features=features,
        spend_col=_SPEND_COL,
        cost_col=_COST_COL,
        conversion_col=_CONVERSION_COL,
        treatment_col=_TREATMENT_COL,
        data=data,
        train_data=train_data,
        distill_data=distill_data,
        test_data=test_data,
    )
