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

"""Tests for the object duplicator."""

from absl.testing import absltest

from fractional_uplift import object_duplicator


class ObjectDuplicatorTest(absltest.TestCase):

  def test_mixin_adds_get_params_method(self):
    class TestClass(object_duplicator.GetParamsMixin, object):

      def __init__(self, my_int: int, my_str: str):
        self.my_int = my_int
        self.my_str = my_str
        self._store_params(my_int=my_int, my_str=my_str)

    input_params = dict(my_int=2, my_str="foo")
    test_object = TestClass(**input_params)
    params = test_object.get_params()

    self.assertDictEqual(params, input_params)

  def test_object_duplicator_makes_copy_of_object(self):
    class TestClass(object_duplicator.GetParamsMixin, object):

      def __init__(self, my_int: int, my_str: str):
        self.my_int = my_int
        self.my_str = my_str
        self._store_params(my_int=my_int, my_str=my_str)

    test_object = TestClass(my_int=2, my_str="foo")
    test_object_copy = object_duplicator.duplicate_object(test_object)

    self.assertIsInstance(test_object_copy, TestClass)
    self.assertIsNot(test_object_copy, test_object)
    self.assertEqual(test_object_copy.my_int, test_object.my_int)
    self.assertEqual(test_object_copy.my_str, test_object.my_str)

  def test_object_duplicator_makes_copy_of_object_in_initial_state(self):
    class TestClass(object_duplicator.GetParamsMixin, object):

      def __init__(self, my_int: int, my_str: str):
        self.my_int = my_int
        self.my_str = my_str
        self._store_params(my_int=my_int, my_str=my_str)

    test_object = TestClass(my_int=2, my_str="foo")
    test_object.my_int = 3
    test_object_copy = object_duplicator.duplicate_object(test_object)

    self.assertEqual(test_object.my_int, 3)
    self.assertEqual(test_object_copy.my_int, 2)


if __name__ == "__main__":
  absltest.main()
