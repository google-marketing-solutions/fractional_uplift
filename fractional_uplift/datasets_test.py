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

"""Tests for all the datasets."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import pandas as pd
import tensorflow as tf

from fractional_uplift import constants
from fractional_uplift import datasets


KPI = constants.KPI
Task = datasets.pandas.Task


class PandasCastColsTest(parameterized.TestCase):

  def test_casts_cols_correctly(self):
    input_data = pd.DataFrame({
        "X1": ["a", "b", "c", "d"],
        "X2": [1, 2, 3, 4],
        "X3": [1.0, 2.0, 3.0, 4.0],
        "X4": [1.0, None, None, 2.0],
    })

    expected_output_data = pd.DataFrame({
        "X1": pd.Categorical(["a", "b", "c", "d"]),
        "X2": [1.0, 2.0, 3.0, 4.0],
        "X3": [1.0, 2.0, 3.0, 4.0],
        "X4": [1.0, np.nan, np.nan, 2.0],
    })

    output_data = datasets.cast_pandas_dataframe_cols(
        input_data,
        cat_feature_cols=["X1"],
        num_feature_cols=["X2", "X3", "X4"],
    )
    pd.testing.assert_frame_equal(output_data, expected_output_data)

  def test_raises_exception_when_cat_feature_cols_overlap_num_feature_cols(
      self,
  ):
    input_data = pd.DataFrame({
        "X1": ["a", "b", "c", "d"],
        "X2": [1, 2, 3, 4],
        "X3": [1.0, 2.0, 3.0, 4.0],
        "X4": [1.0, None, None, 2.0],
    })
    with self.assertRaises(ValueError):
      _ = datasets.cast_pandas_dataframe_cols(
          input_data,
          cat_feature_cols=["X1", "X2"],
          num_feature_cols=["X2", "X3", "X4"],
      )


class PandasDatasetTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.input_features_data = pd.DataFrame({
        "X1": pd.Categorical(["a", "b", "c", "d"]),
        "X2": [1.0, 2.0, 3.0, 4.0],
        "X3": [1.0, 2.0, 3.0, 4.0],
        "X4": [1.0, np.nan, np.nan, 2.0],
    })
    self.input_labels = np.array([1.0, 2.0, 5.0, 2.0])
    self.input_weights = np.array([0.1, 0.2, 0.4, 0.1])

    self.expected_output_data = pd.DataFrame({
        "X1": pd.Categorical(["a", "b", "c", "d"]),
        "X2": [1.0, 2.0, 3.0, 4.0],
        "X3": [1.0, 2.0, 3.0, 4.0],
        "X4": [1.0, np.nan, np.nan, 2.0],
        "label_": pd.Categorical([1.0, 2.0, 5.0, 2.0]),
        "weight_": [0.1, 0.2, 0.4, 0.1],
    })

  def test_sets_features(self):
    data = datasets.PandasDataset(self.input_features_data)
    pd.testing.assert_frame_equal(
        data.as_pd_dataframe(),
        self.expected_output_data[["X1", "X2", "X3", "X4"]],
    )

  def test_adds_categorical_labels_with_correct_name(self):
    data = datasets.PandasDataset(
        self.input_features_data,
        labels=self.input_labels,
        task=Task.CLASSIFICATION,
    )
    pd.testing.assert_frame_equal(
        data.as_pd_dataframe(),
        self.expected_output_data[["X1", "X2", "X3", "X4", "label_"]],
    )

  def test_adds_weights_with_correct_name(self):
    data = datasets.PandasDataset(
        self.input_features_data,
        labels=self.input_labels,
        weights=self.input_weights,
        task=Task.CLASSIFICATION,
    )
    pd.testing.assert_frame_equal(
        data.as_pd_dataframe(), self.expected_output_data
    )

  def test_filters_out_rows_with_zero_weight(self):
    weights = self.input_weights
    weights[0] = 0.0
    data = datasets.PandasDataset(
        self.input_features_data,
        labels=self.input_labels,
        weights=weights,
        task=Task.CLASSIFICATION,
    )
    pd.testing.assert_frame_equal(
        data.as_pd_dataframe(), self.expected_output_data.loc[weights > 0.0]
    )

  @parameterized.parameters([-1.0, None])
  def test_raises_exception_when_weights_are_negative_or_none(self, bad_value):
    weights = self.input_weights
    weights[0] = bad_value

    with self.assertRaises(ValueError):
      datasets.PandasDataset(
          self.input_features_data,
          labels=self.input_labels,
          weights=weights,
          task=Task.CLASSIFICATION,
      )

  def test_raises_exception_when_more_than_3_classes(self):
    with self.assertRaises(ValueError):
      datasets.PandasDataset(
          self.input_features_data,
          labels=np.array(range(4)),
          weights=self.input_weights,
          task=Task.CLASSIFICATION,
      )

  def test_no_exception_when_more_than_3_unique_regression_labels(self):
    data = datasets.PandasDataset(
        self.input_features_data,
        labels=np.array(range(4)),
        weights=self.input_weights,
        task=Task.REGRESSION,
    )
    self.assertIsInstance(data, datasets.PandasDataset)

  def test_raises_exception_when_weights_are_set_but_not_labels(self):
    with self.assertRaises(ValueError):
      datasets.PandasDataset(
          self.input_features_data,
          weights=self.input_weights,
          task=Task.CLASSIFICATION,
      )

  def test_shuffles_rows_when_shuffle_is_true(self):
    data = datasets.PandasDataset(
        self.input_features_data,
        labels=self.input_labels,
        weights=self.input_weights,
        shuffle=True,
        shuffle_seed=13345,
        task=Task.CLASSIFICATION,
    )

    pd.testing.assert_frame_equal(
        data.as_pd_dataframe().sort_index(), self.expected_output_data
    )
    self.assertFalse(data.as_pd_dataframe().equals(self.expected_output_data))

  def test_raises_exception_when_labels_are_wrong_shape(self):
    with self.assertRaises(ValueError):
      datasets.PandasDataset(
          self.input_features_data,
          weights=self.input_weights,
          labels=np.append(self.input_labels, [1.0]),
          task=Task.CLASSIFICATION,
      )

  def test_raises_exception_when_weights_are_wrong_shape(self):
    with self.assertRaises(ValueError):
      datasets.PandasDataset(
          self.input_features_data,
          weights=np.append(self.input_weights, [1.0]),
          labels=self.input_labels,
          task=Task.CLASSIFICATION,
      )

  @parameterized.parameters(["label_", "weight_"])
  def test_raises_exception_with_bad_feature_names(self, bad_name):
    input_features_data = self.input_features_data.copy()
    input_features_data[bad_name] = range(4)
    with self.assertRaises(ValueError):
      datasets.PandasDataset(
          input_features_data,
          weights=self.input_weights,
          labels=self.input_labels,
          task=Task.CLASSIFICATION,
      )

  def test_returns_expected_tf_dataset_features_only(self):
    ds = datasets.PandasDataset(self.input_features_data)

    expected_tf_batch = [{
        "X1": tf.convert_to_tensor(self.expected_output_data["X1"]),
        "X2": tf.convert_to_tensor(self.expected_output_data["X2"]),
        "X3": tf.convert_to_tensor(self.expected_output_data["X3"]),
        "X4": tf.convert_to_tensor(self.expected_output_data["X4"]),
    }]

    self.assertIsInstance(ds.as_tf_dataset(), tf.data.Dataset)
    self.assertEqual(
        str(list(ds.as_tf_dataset().take(4))),
        str(expected_tf_batch),
    )

  def test_returns_expected_tf_dataset_with_labels(self):
    ds = datasets.PandasDataset(
        self.input_features_data,
        labels=self.input_labels,
        task=Task.CLASSIFICATION,
    )

    expected_tf_batch = [(
        {
            "X1": tf.convert_to_tensor(self.expected_output_data["X1"]),
            "X2": tf.convert_to_tensor(self.expected_output_data["X2"]),
            "X3": tf.convert_to_tensor(self.expected_output_data["X3"]),
            "X4": tf.convert_to_tensor(self.expected_output_data["X4"]),
        },
        tf.convert_to_tensor(self.expected_output_data["label_"]),
    )]

    self.assertIsInstance(ds.as_tf_dataset(), tf.data.Dataset)
    self.assertEqual(
        str(list(ds.as_tf_dataset().take(4))),
        str(expected_tf_batch),
    )

  def test_returns_expected_tf_dataset_with_labels_and_weights(self):
    ds = datasets.PandasDataset(
        self.input_features_data,
        labels=self.input_labels,
        weights=self.input_weights,
        task=Task.CLASSIFICATION,
    )

    expected_tf_batch = [(
        {
            "X1": tf.convert_to_tensor(self.expected_output_data["X1"]),
            "X2": tf.convert_to_tensor(self.expected_output_data["X2"]),
            "X3": tf.convert_to_tensor(self.expected_output_data["X3"]),
            "X4": tf.convert_to_tensor(self.expected_output_data["X4"]),
        },
        tf.convert_to_tensor(self.expected_output_data["label_"]),
        tf.convert_to_tensor(self.expected_output_data["weight_"]),
    )]

    self.assertIsInstance(ds.as_tf_dataset(), tf.data.Dataset)
    self.assertEqual(
        str(list(ds.as_tf_dataset().take(4))),
        str(expected_tf_batch),
    )

  def test_regression_task_has_float_label(self):
    ds = datasets.PandasDataset(
        self.input_features_data,
        labels=self.input_labels,
        weights=self.input_weights,
        task=Task.REGRESSION,
    )
    pd.testing.assert_series_equal(
        ds.as_pd_dataframe()["label_"],
        pd.Series(self.input_labels, name="label_", dtype=np.float64),
    )

  def test_raises_exception_if_label_set_without_task(self):
    with self.assertRaises(ValueError):
      datasets.PandasDataset(
          self.input_features_data,
          labels=self.input_labels,
      )

  def test_len_returns_number_of_rows(self):
    ds = datasets.PandasDataset(self.input_features_data)
    self.assertEqual(len(ds), 4)

  def test_labels_are_constant_returns_true_if_constant(self):
    ds = datasets.PandasDataset(
        self.input_features_data, labels=np.ones(4), task=Task.REGRESSION
    )
    self.assertTrue(ds.labels_are_constant())
    self.assertEqual(ds.label_average(), 1.0)

  def test_labels_are_constant_returns_false_if_not_constant_for_regression(
      self,
  ):
    ds = datasets.PandasDataset(
        self.input_features_data, labels=self.input_labels, task=Task.REGRESSION
    )
    self.assertFalse(ds.labels_are_constant())
    self.assertEqual(ds.label_average(), np.mean(self.input_labels))

  def test_labels_are_constant_returns_false_if_not_constant_for_classification(
      self,
  ):
    ds = datasets.PandasDataset(
        self.input_features_data,
        labels=self.input_labels,
        task=Task.CLASSIFICATION,
    )
    self.assertFalse(ds.labels_are_constant())
    self.assertEqual(ds.label_average(), 2.0)

  def test_can_get_params_from_dataset(self):
    input_params = dict(
        features_data=self.input_features_data,
        labels=self.input_labels,
        weights=self.input_weights,
        shuffle=False,
        shuffle_seed=1234,
        copy=True,
        task=Task.CLASSIFICATION,
    )
    ds = datasets.PandasDataset(**input_params)

    output_params = ds.get_params()

    self.assertEqual(output_params, input_params)


class PandasTrainDataTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.input_features_data = pd.DataFrame({
        "X1": pd.Categorical(["a", "b", "c", "d"]),
        "X2": [1.0, 2.0, 3.0, 4.0],
        "X3": [1.0, 2.0, 3.0, 4.0],
        "X4": [1.0, np.nan, np.nan, 2.0],
    })
    self.input_maximize_kpi = np.array([1.0, 2.0, 3.0, 4.0])
    self.input_constraint_kpi = np.array([5.0, 6.0, 7.0, 8.0])
    self.input_constraint_offset_kpi = np.array([9.0, 10.0, 11.0, 12.0])
    self.input_is_treated = np.array([0, 0, 1, 1])
    self.input_treatment_propensity = np.array([0.5, 0.2, 0.5, 0.2])
    self.input_shuffle_seed = 12345
    self.sample_weight = np.array([0.2, 1.0, 2.0, 1.5])

    self.expected_inverse_propensity_weights = np.array([2.0, 1.25, 2.0, 5.0])

    self.expected_inc_classifier_data = dict()
    for kpi_name, kpi_value in zip(
        list(KPI),
        [
            self.input_maximize_kpi,
            self.input_constraint_kpi,
            self.input_constraint_offset_kpi,
        ],
    ):
      self.expected_inc_classifier_data[kpi_name.name] = datasets.PandasDataset(
          self.input_features_data,
          labels=self.input_is_treated,
          weights=kpi_value * self.expected_inverse_propensity_weights,
          shuffle=True,
          shuffle_seed=self.input_shuffle_seed,
          task=Task.CLASSIFICATION,
      )

    self.expected_single_treatment_data = dict()
    for kpi_name, kpi_value in zip(
        list(KPI),
        [
            self.input_maximize_kpi,
            self.input_constraint_kpi,
            self.input_constraint_offset_kpi,
        ],
    ):
      for treatment in [0, 1]:
        key = kpi_name.name + f"_{treatment}"
        mask = self.input_is_treated == treatment
        self.expected_single_treatment_data[key] = datasets.PandasDataset(
            self.input_features_data.loc[mask],
            labels=kpi_value[mask],
            weights=self.expected_inverse_propensity_weights[mask],
            shuffle=True,
            shuffle_seed=self.input_shuffle_seed,
            task=Task.REGRESSION,
        )

  def test_inverse_propensity_weights_are_correct(self):
    train_data = datasets.PandasTrainData(
        self.input_features_data,
        maximize_kpi=self.input_maximize_kpi,
        is_treated=self.input_is_treated,
        treatment_propensity=self.input_treatment_propensity,
        shuffle_seed=self.input_shuffle_seed,
    )

    np.testing.assert_array_equal(
        train_data.inverse_propensity_weights,
        self.expected_inverse_propensity_weights,
    )

  @parameterized.parameters(list(KPI))
  def test_can_check_if_kpi_is_non_negative(self, negative_kpi):
    kpis = dict(
        maximize_kpi=self.input_maximize_kpi,
        constraint_kpi=self.input_constraint_kpi,
        constraint_offset_kpi=self.input_constraint_offset_kpi,
    )
    kpis[negative_kpi.name.lower()] *= -1.0

    train_data = datasets.PandasTrainData(
        self.input_features_data,
        is_treated=self.input_is_treated,
        treatment_propensity=self.input_treatment_propensity,
        shuffle_seed=self.input_shuffle_seed,
        **kpis,
    )
    for kpi in KPI:
      if kpi == negative_kpi:
        self.assertFalse(train_data.has_non_negative_kpi(kpi))
      else:
        self.assertTrue(train_data.has_non_negative_kpi(kpi))

  def test_construct_maximize_kpi_constraint_kpi_constraint_offset_kpi_data(
      self,
  ):
    train_data = datasets.PandasTrainData(
        self.input_features_data,
        maximize_kpi=self.input_maximize_kpi,
        constraint_kpi=self.input_constraint_kpi,
        constraint_offset_kpi=self.input_constraint_offset_kpi,
        is_treated=self.input_is_treated,
        treatment_propensity=self.input_treatment_propensity,
        shuffle_seed=self.input_shuffle_seed,
    )
    self.assertTrue(train_data.has_kpi(KPI.MAXIMIZE_KPI))
    self.assertTrue(train_data.has_kpi(KPI.CONSTRAINT_KPI))
    self.assertTrue(train_data.has_kpi(KPI.CONSTRAINT_OFFSET_KPI))

    for kpi in KPI:
      self.assertIsInstance(
          train_data.get_inc_classifier_data(kpi),
          datasets.PandasDataset,
      )
      pd.testing.assert_frame_equal(
          train_data.get_inc_classifier_data(kpi).as_pd_dataframe(),
          self.expected_inc_classifier_data[kpi.name].as_pd_dataframe(),
      )

      for treatment in [True, False]:
        self.assertIsInstance(
            train_data.get_data_for_single_treatment(kpi, treatment),
            datasets.PandasDataset,
        )
        pd.testing.assert_frame_equal(
            train_data.get_data_for_single_treatment(
                kpi, treatment
            ).as_pd_dataframe(),
            self.expected_single_treatment_data[
                kpi.name + f"_{int(treatment)}"
            ].as_pd_dataframe(),
        )

    expected_kpi_weight_dataset = datasets.PandasDataset(
        pd.concat([self.input_features_data] * 3).copy().reset_index(drop=True),
        labels=np.array(4 * [0] + 4 * [1] + 4 * [2]),
        weights=np.concatenate([
            self.input_maximize_kpi * self.expected_inverse_propensity_weights,
            self.input_constraint_kpi
            * self.expected_inverse_propensity_weights,
            self.input_constraint_offset_kpi
            * self.expected_inverse_propensity_weights,
        ]),
        shuffle=True,
        shuffle_seed=self.input_shuffle_seed,
        task=Task.CLASSIFICATION,
    )
    self.assertIsInstance(
        train_data.get_kpi_weight_data(), datasets.PandasDataset
    )

    pd.testing.assert_frame_equal(
        train_data.get_kpi_weight_data().as_pd_dataframe(),
        expected_kpi_weight_dataset.as_pd_dataframe(),
    )

  def test_construct_maximize_kpi_constraint_kpi_data(self):
    train_data = datasets.PandasTrainData(
        self.input_features_data,
        maximize_kpi=self.input_maximize_kpi,
        constraint_kpi=self.input_constraint_kpi,
        is_treated=self.input_is_treated,
        treatment_propensity=self.input_treatment_propensity,
        shuffle_seed=self.input_shuffle_seed,
    )
    self.assertTrue(train_data.has_kpi(KPI.MAXIMIZE_KPI))
    self.assertTrue(train_data.has_kpi(KPI.CONSTRAINT_KPI))
    self.assertFalse(train_data.has_kpi(KPI.CONSTRAINT_OFFSET_KPI))

    self.assertIsNone(
        train_data.get_inc_classifier_data(KPI.CONSTRAINT_OFFSET_KPI)
    )
    self.assertIsNone(
        train_data.get_data_for_single_treatment(
            KPI.CONSTRAINT_OFFSET_KPI, False
        )
    )
    self.assertIsNone(
        train_data.get_data_for_single_treatment(
            KPI.CONSTRAINT_OFFSET_KPI, True
        )
    )

    for kpi in [KPI.MAXIMIZE_KPI, KPI.CONSTRAINT_KPI]:
      self.assertIsInstance(
          train_data.get_inc_classifier_data(kpi),
          datasets.PandasDataset,
      )
      pd.testing.assert_frame_equal(
          train_data.get_inc_classifier_data(kpi).as_pd_dataframe(),
          self.expected_inc_classifier_data[kpi.name].as_pd_dataframe(),
      )

      for treatment in [True, False]:
        self.assertIsInstance(
            train_data.get_data_for_single_treatment(kpi, treatment),
            datasets.PandasDataset,
        )
        pd.testing.assert_frame_equal(
            train_data.get_data_for_single_treatment(
                kpi, treatment
            ).as_pd_dataframe(),
            self.expected_single_treatment_data[
                kpi.name + f"_{int(treatment)}"
            ].as_pd_dataframe(),
        )

    expected_kpi_weight_dataset = datasets.PandasDataset(
        pd.concat([self.input_features_data] * 2).copy().reset_index(drop=True),
        labels=np.array(4 * [0] + 4 * [1]),
        weights=np.concatenate([
            self.input_maximize_kpi * self.expected_inverse_propensity_weights,
            self.input_constraint_kpi
            * self.expected_inverse_propensity_weights,
        ]),
        shuffle=True,
        shuffle_seed=self.input_shuffle_seed,
        task=Task.CLASSIFICATION,
    )
    self.assertIsInstance(
        train_data.get_kpi_weight_data(), datasets.PandasDataset
    )

    pd.testing.assert_frame_equal(
        train_data.get_kpi_weight_data().as_pd_dataframe(),
        expected_kpi_weight_dataset.as_pd_dataframe(),
    )

  def test_construct_maximize_kpi_data(self):
    train_data = datasets.PandasTrainData(
        self.input_features_data,
        maximize_kpi=self.input_maximize_kpi,
        is_treated=self.input_is_treated,
        treatment_propensity=self.input_treatment_propensity,
        shuffle_seed=self.input_shuffle_seed,
    )
    self.assertTrue(train_data.has_kpi(KPI.MAXIMIZE_KPI))
    self.assertFalse(train_data.has_kpi(KPI.CONSTRAINT_KPI))
    self.assertFalse(train_data.has_kpi(KPI.CONSTRAINT_OFFSET_KPI))

    for kpi in [
        KPI.CONSTRAINT_KPI,
        KPI.CONSTRAINT_OFFSET_KPI,
    ]:
      self.assertIsNone(train_data.get_inc_classifier_data(kpi))
      self.assertIsNone(train_data.get_data_for_single_treatment(kpi, False))
      self.assertIsNone(train_data.get_data_for_single_treatment(kpi, True))

    self.assertIsInstance(
        train_data.get_inc_classifier_data(KPI.MAXIMIZE_KPI),
        datasets.PandasDataset,
    )
    pd.testing.assert_frame_equal(
        train_data.get_inc_classifier_data(KPI.MAXIMIZE_KPI).as_pd_dataframe(),
        self.expected_inc_classifier_data[
            KPI.MAXIMIZE_KPI.name
        ].as_pd_dataframe(),
    )

    for treatment in [True, False]:
      self.assertIsInstance(
          train_data.get_data_for_single_treatment(KPI.MAXIMIZE_KPI, treatment),
          datasets.PandasDataset,
      )
      pd.testing.assert_frame_equal(
          train_data.get_data_for_single_treatment(
              KPI.MAXIMIZE_KPI, treatment
          ).as_pd_dataframe(),
          self.expected_single_treatment_data[
              KPI.MAXIMIZE_KPI.name + f"_{int(treatment)}"
          ].as_pd_dataframe(),
      )

    self.assertIsNone(train_data.get_kpi_weight_data())

  def test_data_is_cached(self):
    train_data = datasets.PandasTrainData(
        self.input_features_data,
        maximize_kpi=self.input_maximize_kpi,
        constraint_kpi=self.input_constraint_kpi,
        constraint_offset_kpi=self.input_constraint_offset_kpi,
        is_treated=self.input_is_treated,
        treatment_propensity=self.input_treatment_propensity,
        shuffle_seed=self.input_shuffle_seed,
    )

    for kpi in KPI:
      self.assertIs(
          train_data.get_inc_classifier_data(kpi),
          train_data.get_inc_classifier_data(kpi),
      )

      for treatment in [True, False]:
        self.assertIs(
            train_data.get_data_for_single_treatment(kpi, treatment),
            train_data.get_data_for_single_treatment(kpi, treatment),
        )

    self.assertIs(
        train_data.get_kpi_weight_data(),
        train_data.get_kpi_weight_data(),
    )

  def test_raises_exception_if_constraint_offset_kpi_set_but_not_constraint_kpi(
      self,
  ):
    with self.assertRaises(ValueError):
      datasets.PandasTrainData(
          self.input_features_data,
          maximize_kpi=self.input_maximize_kpi,
          constraint_offset_kpi=self.input_constraint_offset_kpi,
          is_treated=self.input_is_treated,
          treatment_propensity=self.input_treatment_propensity,
          shuffle_seed=self.input_shuffle_seed,
      )

  @parameterized.parameters([2, 0.5, -1])
  def test_raises_exception_if_is_treated_not_binary(self, bad_value):
    is_treated = self.input_is_treated.copy()
    is_treated = is_treated.astype(type(bad_value))
    is_treated[0] = bad_value

    with self.assertRaises(ValueError):
      datasets.PandasTrainData(
          self.input_features_data,
          maximize_kpi=self.input_maximize_kpi,
          constraint_kpi=self.input_constraint_kpi,
          constraint_offset_kpi=self.input_constraint_offset_kpi,
          is_treated=is_treated,
          treatment_propensity=self.input_treatment_propensity,
          shuffle_seed=self.input_shuffle_seed,
      )

  @parameterized.parameters([1.1, 1.0, 0.0, -0.1])
  def test_raises_exception_if_treatment_propensity_not_valid_probability(
      self, bad_value
  ):
    treatment_propensity = self.input_treatment_propensity.copy()
    treatment_propensity[0] = bad_value

    with self.assertRaises(ValueError):
      datasets.PandasTrainData(
          self.input_features_data,
          maximize_kpi=self.input_maximize_kpi,
          constraint_kpi=self.input_constraint_kpi,
          constraint_offset_kpi=self.input_constraint_offset_kpi,
          is_treated=self.input_is_treated,
          treatment_propensity=treatment_propensity,
          shuffle_seed=self.input_shuffle_seed,
      )

  @parameterized.parameters([-0.5, -1.0])
  def test_raises_exception_if_sample_weights_are_negative(self, bad_value):
    sample_weight = self.sample_weight.copy()
    sample_weight[0] = bad_value

    with self.assertRaises(ValueError):
      datasets.PandasTrainData(
          self.input_features_data,
          maximize_kpi=self.input_maximize_kpi,
          constraint_kpi=self.input_constraint_kpi,
          constraint_offset_kpi=self.input_constraint_offset_kpi,
          is_treated=self.input_is_treated,
          treatment_propensity=self.input_treatment_propensity,
          shuffle_seed=self.input_shuffle_seed,
          sample_weight=sample_weight,
      )

  @parameterized.parameters([
      KPI.MAXIMIZE_KPI,
      KPI.CONSTRAINT_KPI,
      KPI.CONSTRAINT_OFFSET_KPI,
  ])
  def test_inc_classifier_rows_are_weighed_by_sample_weights(self, kpi):
    weighted_train_data = datasets.PandasTrainData(
        self.input_features_data,
        maximize_kpi=self.input_maximize_kpi,
        constraint_kpi=self.input_constraint_kpi,
        constraint_offset_kpi=self.input_constraint_offset_kpi,
        is_treated=self.input_is_treated,
        treatment_propensity=self.input_treatment_propensity,
        shuffle_seed=self.input_shuffle_seed,
        sample_weight=self.sample_weight,
    )
    unweighted_train_data = datasets.PandasTrainData(
        self.input_features_data,
        maximize_kpi=self.input_maximize_kpi,
        constraint_kpi=self.input_constraint_kpi,
        constraint_offset_kpi=self.input_constraint_offset_kpi,
        is_treated=self.input_is_treated,
        treatment_propensity=self.input_treatment_propensity,
        shuffle_seed=self.input_shuffle_seed,
    )

    weighted_weights = weighted_train_data.get_inc_classifier_data(
        kpi
    ).as_pd_dataframe()["weight_"]
    unweighted_weights = unweighted_train_data.get_inc_classifier_data(
        kpi
    ).as_pd_dataframe()["weight_"]

    np.testing.assert_allclose(
        np.sort(weighted_weights / unweighted_weights),
        np.sort(self.sample_weight),
    )

  @parameterized.product(
      kpi=[
          KPI.MAXIMIZE_KPI,
          KPI.CONSTRAINT_KPI,
          KPI.CONSTRAINT_OFFSET_KPI,
      ],
      is_treated=[True, False],
  )
  def test_single_treatment_data_rows_are_weighed_by_sample_weights(
      self, kpi, is_treated
  ):
    weighted_train_data = datasets.PandasTrainData(
        self.input_features_data,
        maximize_kpi=self.input_maximize_kpi,
        constraint_kpi=self.input_constraint_kpi,
        constraint_offset_kpi=self.input_constraint_offset_kpi,
        is_treated=self.input_is_treated,
        treatment_propensity=self.input_treatment_propensity,
        shuffle_seed=self.input_shuffle_seed,
        sample_weight=self.sample_weight,
    )
    unweighted_train_data = datasets.PandasTrainData(
        self.input_features_data,
        maximize_kpi=self.input_maximize_kpi,
        constraint_kpi=self.input_constraint_kpi,
        constraint_offset_kpi=self.input_constraint_offset_kpi,
        is_treated=self.input_is_treated,
        treatment_propensity=self.input_treatment_propensity,
        shuffle_seed=self.input_shuffle_seed,
    )

    weighted_weights = weighted_train_data.get_data_for_single_treatment(
        kpi, is_treated
    ).as_pd_dataframe()["weight_"]
    unweighted_weights = unweighted_train_data.get_data_for_single_treatment(
        kpi, is_treated
    ).as_pd_dataframe()["weight_"]

    expected_sample_weights = self.sample_weight[
        self.input_is_treated == int(is_treated)
    ]

    np.testing.assert_allclose(
        np.sort(weighted_weights / unweighted_weights),
        np.sort(expected_sample_weights),
    )

  def test_kpi_weight_data_rows_are_weighed_by_sample_weights(self):
    weighted_train_data = datasets.PandasTrainData(
        self.input_features_data,
        maximize_kpi=self.input_maximize_kpi,
        constraint_kpi=self.input_constraint_kpi,
        constraint_offset_kpi=self.input_constraint_offset_kpi,
        is_treated=self.input_is_treated,
        treatment_propensity=self.input_treatment_propensity,
        shuffle_seed=self.input_shuffle_seed,
        sample_weight=self.sample_weight,
    )
    unweighted_train_data = datasets.PandasTrainData(
        self.input_features_data,
        maximize_kpi=self.input_maximize_kpi,
        constraint_kpi=self.input_constraint_kpi,
        constraint_offset_kpi=self.input_constraint_offset_kpi,
        is_treated=self.input_is_treated,
        treatment_propensity=self.input_treatment_propensity,
        shuffle_seed=self.input_shuffle_seed,
    )

    weighted_weights = (
        weighted_train_data.get_kpi_weight_data().as_pd_dataframe()["weight_"]
    )
    unweighted_weights = (
        unweighted_train_data.get_kpi_weight_data().as_pd_dataframe()["weight_"]
    )

    expected_sample_weights = np.concatenate([self.sample_weight] * 3)

    np.testing.assert_allclose(
        np.sort(weighted_weights / unweighted_weights),
        np.sort(expected_sample_weights),
    )


if __name__ == "__main__":
  absltest.main()
