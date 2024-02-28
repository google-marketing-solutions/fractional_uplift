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

"""A module containing all the base models that can be used.

Base models are the fundamental ML algorithms used by the FractionalUpliftModel
to learn the uplift. These can be any ML classification algorithm, as long as 
it returns a score that can be interpreted as the probability of a sample
being assigned to a class.
"""

from fractional_uplift.base_models import _base
from fractional_uplift.base_models import tf_decision_forests


BaseModel = _base.BaseModel
TensorflowDecisionForestClassifier = (
    tf_decision_forests.TensorflowDecisionForestClassifier
)
TensorflowDecisionForestRegressor = (
    tf_decision_forests.TensorflowDecisionForestRegressor
)
