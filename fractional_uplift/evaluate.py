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

"""Module containing tools to evaluate uplift models."""

from typing import List

import numpy as np
import pandas as pd

from fractional_uplift import constants


EffectType = constants.EffectType


class TemporaryDataframeCopy:
  """A context manager to create a temporary copy of a pandas dataframe."""

  def __init__(self, data: pd.DataFrame):
    self.data_copy = data.copy()

  def __enter__(self):
    return self.data_copy

  def __exit__(self, exc_type, exc_val, exc_tb):
    del self.data_copy


class UpliftEvaluator:
  """Calculates evaluation metrics for uplift models.

  It orders descending all the rows in the test dataframe by the uplift model
  score, and then groups them into bins. It then calculates the cumulative
  sum of the user specified metrics in metric_cols separately for the
  treatment group and the control group, based on the is_treated_col.

  The metric sums are adjusted by the treatment propensity, and then the
  difference between treatment and control is calculated for all metrics:
  this is the incremenality. If the treatment propensity is correct, then the
  estimates of the incrementality is unbiased even if the dataset is not
  generated from a randomised A/B test.

  The correction for the treatment propensity (tp) depends on the effect type
  specified:

  If effect_type = EffectType.ATE
    Samples in control are weighed by 1 / (1 - tp)
    Samples in treatment are weighed by 1 / tp
    This estimates the average treatment effect across both the control and
    treated samples.

  If effect_type = EffectType.ATT
    Samples in control are weighed by tp / (1 - tp)
    Samples in treatment are unweighted
    This estimates the average treatment effect across only the treated
    samples.

  If effect_type = EffectType.ATC
    Samples in control are unweighted
    Samples in treatment are weighed by (1 - tp) / tp
    This estimates the average treatment effect across only the control
    samples.
  """

  _results_lst: List[pd.DataFrame]

  AVG_SCORE_COL: str = "avg_score"
  SHARE_TARGETED_COL: str = "share_targeted"
  RANDOM_SCORE_COL: str = "random"
  DUMMY_COL: str = "dummy_"

  def __init__(
      self,
      is_treated_col: str,
      treatment_propensity_col: str,
      metric_cols: List[str],
      n_bins: int = 100,
      effect_type: EffectType = EffectType.ATE,
  ):
    """Initialises the evaluator.

    Args:
      is_treated_col: The column containing the treatment assignment
      treatment_propensity_col: The column containing the treatment propensity
      metric_cols: A list of metrics to calculate the cumulative incrementality
      n_bins: The number of bins to break the score into for calculation
      effect_type: The type of incrementality to estimate.
    """
    bad_metric_col_names = {
        self.SHARE_TARGETED_COL,
        self.AVG_SCORE_COL,
        self.RANDOM_SCORE_COL,
        self.DUMMY_COL,
    } & set(metric_cols + [is_treated_col, treatment_propensity_col])
    if bad_metric_col_names:
      raise ValueError(
          f"Must not use the following column names: {bad_metric_col_names}"
      )

    self._effect_type = effect_type
    self._n_bins = n_bins
    self._is_treated_col = is_treated_col
    self._metric_cols = metric_cols
    self._treatment_propensity_col = treatment_propensity_col

  def _calculate_inverse_propensity_weights(
      self, data: pd.DataFrame
  ) -> pd.Series:
    """Calculates the inverse propensity weights and adds them to data.

    The weights are calculated from the treatment propensity (tp), based on the
    effect type:

    If effect_type = EffectType.ATE
      Samples in control are weighed by 1 / (1 - tp)
      Samples in treatment are weighed by 1 / tp
      This estimates the average treatment effect across both the control and
      treated samples.

    If effect_type = EffectType.ATT
      Samples in control are weighed by tp / (1 - tp)
      Samples in treatment are unweighted
      This estimates the average treatment effect across only the treated
      samples.

    If effect_type = EffectType.ATC
      Samples in control are unweighted
      Samples in treatment are weighed by (1 - tp) / tp
      This estimates the average treatment effect across only the control
      samples.

    Args:
      data: The data to calculated the weights for

    Returns:
      The data with a column named "inverse_propensity_weights"

    Raises:
      RuntimeError: If self._effect_type is unexpected.
    """
    if self._effect_type == EffectType.ATE:
      # Average treatment effect across all samples
      inverse_propensity_weights = data[self._is_treated_col] / data[
          self._treatment_propensity_col
      ] + (1.0 - data[self._is_treated_col]) / (
          1.0 - data[self._treatment_propensity_col]
      )
    elif self._effect_type == EffectType.ATC:
      # Average treatment effect across control samples
      inverse_propensity_weights = (
          1.0 - data[self._treatment_propensity_col]
      ) * data[self._is_treated_col] / data[self._treatment_propensity_col] + (
          1.0 - data[self._is_treated_col]
      )
    elif self._effect_type == EffectType.ATT:
      # Average treatment effect across treated samples
      inverse_propensity_weights = data[self._is_treated_col] + data[
          self._treatment_propensity_col
      ] * (1.0 - data[self._is_treated_col]) / (
          1.0 - data[self._treatment_propensity_col]
      )
    else:
      raise RuntimeError("The effect type is unexpected.")

    return inverse_propensity_weights

  def _calculate_totals_per_score_bin(
      self,
      data: pd.DataFrame,
      score_col: str,
      inverse_propensity_weights: pd.Series,
  ) -> pd.DataFrame:
    """Calculates the totals of all the metrics, per score bin.
    
    Args:
      data: The data to perform the calculation on.
      score_col: The score column to sort by before binning.
      inverse_propensity_weights: The weights for each row in the data
      
    Returns:
      A dataframe with one row per score bin, and 2 columns per metric - the
      total of that metric in control / treatment.
    """
    data[self.DUMMY_COL] = np.random.rand(len(data.index.values))
    data = data.sort_values([score_col, self.DUMMY_COL], ascending=False)
    data[self.SHARE_TARGETED_COL] = (
        np.floor(
            0.9999999
            * self._n_bins
            * np.arange(len(data.index.values))
            / len(data.index.values)
        )
        / self._n_bins
        + 1.0 / self._n_bins
    )
    data[self.AVG_SCORE_COL] = data.groupby(self.SHARE_TARGETED_COL)[
        score_col
    ].transform(
        lambda rows: np.average(
            rows, weights=inverse_propensity_weights.loc[rows.index]
        )
    )

    out_data = pd.pivot_table(
        data,
        values=self._metric_cols,
        index=[self.SHARE_TARGETED_COL, self.AVG_SCORE_COL],
        columns=self._is_treated_col,
        aggfunc=lambda rows: np.sum(
            rows * inverse_propensity_weights.loc[rows.index]
        ),
        fill_value=0.0,
    )

    return out_data.reset_index()

  def _calculate_random_totals_per_score_bin(
      self, data: pd.DataFrame, inverse_propensity_weights: pd.Series
  ) -> pd.DataFrame:
    """Calculates the expected total per score bin, if the score is random.
    
    Args:
      data: The data to perform the calculation on.
      inverse_propensity_weights: The weights for each row in the data
      
    Returns:
      A dataframe with one row per random score bin, and 2 columns per metric - 
      the total of that metric in control / treatment.
    """
    data[self.DUMMY_COL] = 0
    baseline_totals = pd.pivot_table(
        data,
        index=self.DUMMY_COL,
        values=self._metric_cols,
        columns=self._is_treated_col,
        aggfunc=lambda rows: np.sum(
            rows * inverse_propensity_weights.loc[rows.index]
        ),
    )
    out_data = pd.concat([baseline_totals] * self._n_bins).reset_index(
        drop=True
    )
    out_data /= self._n_bins

    out_data[self.SHARE_TARGETED_COL] = np.linspace(
        1.0 / self._n_bins, 1.0, self._n_bins
    )
    out_data[self.AVG_SCORE_COL] = 0.5

    return out_data

  def _calculate_incrementality_per_score_bin(
      self,
      data: pd.DataFrame,
      score_col: str,
      inverse_propensity_weights: pd.Series,
  ) -> pd.DataFrame:
    """Calculates the incrementality of the metrics, for each score bin.
    
    Args:
      data: The data to perform the calculation on.
      score_col: The score column to sort by before binning.
      inverse_propensity_weights: The weights for each row in the data
      
    Returns:
      A dataframe with one row per score bin, and 3 columns per metric - the
      total of that metric in control / treatment and the incrementality. It 
      also has two more columns - the average score per bin, and the cumulative
      share of rows targeted.
    """
    if score_col == self.RANDOM_SCORE_COL:
      out_data = self._calculate_random_totals_per_score_bin(
          data, inverse_propensity_weights
      )
    else:
      out_data = self._calculate_totals_per_score_bin(
          data, score_col, inverse_propensity_weights
      )

    out_cols = [self.SHARE_TARGETED_COL, self.AVG_SCORE_COL]
    for metric in self._metric_cols:
      out_data[f"{metric}__inc"] = out_data[metric][1] - out_data[metric][0]
      out_data[f"{metric}__control"] = out_data[metric][0]
      out_data[f"{metric}__treated"] = out_data[metric][1]
      out_cols.extend(
          [f"{metric}__inc", f"{metric}__control", f"{metric}__treated"]
      )

    out_data = out_data[out_cols].sort_values(self.SHARE_TARGETED_COL)
    out_data.columns = out_data.columns.get_level_values(0)

    return out_data

  def _calculate_cumulative_incrementality(
      self, data: pd.DataFrame
  ) -> pd.DataFrame:
    """Calculates the cumulative metrics.
    
    Args:
      data: The data containing the metrics per score bin.
      
    Returns:
      The data with the cumulative metrics added. All the cumulative metrics
      end with _cum.
    """
    for metric in self._metric_cols:
      data[f"{metric}__inc_cum"] = data[f"{metric}__inc"].cumsum()
      data[f"{metric}__control_cum"] = data[f"{metric}__control"].cumsum()
      data[f"{metric}__treated_cum"] = data[f"{metric}__treated"].cumsum()

    return data

  def _add_metadata(
      self, data: pd.DataFrame, name: str
  ) -> pd.DataFrame:
    """Adds the score name to the dataframe.
    
    Args:
      data: The data to add the name to.
      name: The name of the score.
    """
    data["name"] = name
    return data

  def _calculate_composite_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
    """Calculates complex metrics.
    
    This is where the user can calculate complex metrics which are composed of 
    other metrics. A good example is RoI, which is the incremental revenue
    divided by incremental cost.
    
    By default this does nothing, to use it the user must subclass this 
    class and overwrite this method.
    """
    return data

  def _analyze_score(
      self,
      data: pd.DataFrame,
      score_col: str,
      inverse_propensity_weights: pd.Series,
  ) -> pd.DataFrame:
    """Performs the complete analysis for a single score col.
    
    Args:
      data: The data to use for the calculation.
      score_col: The score to be evaluated.
      inverse_propensity_weights: The weights for each row in the data.
      
    Returns:
      A dataframe with the cumulative metrics per score col bin.
    """
    with TemporaryDataframeCopy(data) as tmp_data:
      out_data = (
          tmp_data.pipe(
              self._calculate_incrementality_per_score_bin,
              score_col,
              inverse_propensity_weights,
          )
          .pipe(self._calculate_cumulative_incrementality)
          .pipe(self._add_metadata, score_col)
          .pipe(self._calculate_composite_metrics)
      )
      return out_data

  def evaluate(
      self,
      test_data: pd.DataFrame,
      score_cols: List[str],
  ) -> pd.DataFrame:
    """Evaluates a set of different scores on the same data.

    Args:
      test_data: The data to use for the evaluation..
      score_cols: The columns in the test data containing the model scores to be
        evaluated.

    Returns:
      A dataframe with the cumulative metrics per score col bin, for each score
      evaluated. If evaluating a single metric, metric_1, the output columns
      would be:
        - name: The name of the score being evaluated.
        - share_targeted (the cumulative share of all rows)
        - avg_score (the average score in that bin)
        - metric_1__control (the total for metric 1 in control in that bin)
        - metric_1__control_cum (the cumulative total for metric 1 in control
            up to and including that bin)
        - metric_1__treated (the total for metric 1 in control in that bin)
        - metric_1__treated_cum (the cumulative total for metric 1 in control
            up to and including that bin)
        - metric_1__inc (the incrementality of metric 1 in that bin)
        - metric_1__inc_cum (the cumulative incrementality of metric 1 up to
            and including that bin)

      The dataframe will have n_bins * (n_score_cols + 1) rows.
    """
    with TemporaryDataframeCopy(test_data) as tmp_data:
      inverse_propensity_weights = self._calculate_inverse_propensity_weights(
          tmp_data
      )
      results_lst = [
          self._analyze_score(tmp_data, score_col, inverse_propensity_weights)
          for score_col in score_cols
      ]
      results_lst.append(
          self._analyze_score(
              tmp_data, self.RANDOM_SCORE_COL, inverse_propensity_weights
          )
      )

    return pd.concat(results_lst).reset_index(drop=True)


def calculate_auc(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    sort_col: str | None = None,
) -> float:
  """Calculates the area under the curve (AUC).

  Args:
    data: The data to be used for the calculation.
    x_col: The column name of the x variable.
    y_col: The column name of the y variable.
    sort_col: The column to sort by when calculating the AUC. If None, sorts by
      the x_col.

  Returns:
    The integral of y vs x, ordered by the sort col.
  """
  with TemporaryDataframeCopy(data) as tmp_data:
    sort_col = sort_col or x_col
    tmp_data = tmp_data.sort_values(sort_col)
    return np.trapz(tmp_data[y_col].values, tmp_data[x_col].values)
