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

"""Classes used to make duplicates of objects.

These are helper methods that are used for a variety of different things
throughout the package where duplicating an object is needed.
"""

import inspect
from typing import Any, Dict, List, Protocol


class GetParamsMixin:
  """A mixin to provide the functionality for get_params()."""

  _init_params: Dict[str, Any]

  @classmethod
  def _get_param_names(cls) -> List[str]:
    """Gets the parameter names for the object.

    Adapted from scikit-learn:
    https://github.com/scikit-learn/scikit-learn/blob/9aaed4987/sklearn/base.py

    Returns:
      List of parameter names.

    Raises:
      RuntimeError: If the class has varargs.
    """
    if cls.__init__ is object.__init__:
      # No explicit constructor to introspect
      return []

    # introspect the constructor arguments to find the model parameters
    # to represent
    init_signature = inspect.signature(cls.__init__)
    # Consider the constructor parameters excluding 'self'
    parameters = [
        param
        for param in init_signature.parameters.values()
        if param.name != "self" and param.kind != param.VAR_KEYWORD
    ]
    for param in parameters:
      if param.kind == param.VAR_POSITIONAL:
        raise RuntimeError(
            "Base models should always "
            "specify their parameters in the signature"
            " of their __init__ (no varargs)."
            " %s with constructor %s doesn't "
            " follow this convention." % (cls, init_signature)
        )
    # Extract and sort argument names excluding 'self'
    return sorted([param.name for param in parameters])

  def _store_params(self, **params: Any):
    """Stores the parameters passed to __init__.

    First the parameters are validated against the signature of __init__,
    to check that they are correct. Then they are copied and stored, so they
    can be retrieved later.
    """
    expected_params = self._get_param_names()
    if set(expected_params) != set(params.keys()):
      raise RuntimeError(
          f"Params do not match signature. Signature: {expected_params}. "
          f"params: {params.keys()}"
      )
    self._init_params = params.copy()

  def get_params(self) -> Dict[str, Any]:
    """Gets the parameters for this model.

    Returns:
        Parameter names mapped to their values.
    """
    return self._init_params.copy()


class GetParamsProtocol(Protocol):
  """A protocol for an object with a get_params method."""

  def get_params(self) -> Dict[str, Any]:
    ...


def duplicate_object(obj: GetParamsProtocol, **overwrite_params: Any) -> Any:
  """Duplicates the object in its initial state.

  Makes a copy of the object, with the same parameters passed to the
  constructor.

  Args:
    obj: The object to duplicate.
    **overwrite_params: Parameters to overwrite from the original object when
      creating the new one.

  Returns:
    A duplicate of the object.
  """

  params = obj.get_params() | overwrite_params
  return obj.__class__(**params)
