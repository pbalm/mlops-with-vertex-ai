# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Model features metadata utils."""


# FEATURE_NAMES = [
#     "trip_month",
#     "trip_day",
#     "trip_day_of_week",
#     "trip_hour",
#     "trip_seconds",
#     "trip_miles",
#     "payment_type",
#     "pickup_grid",
#     "dropoff_grid",
#     "euclidean",
#     "loc_cross",
# ]

TARGET_FEATURE_NAME = "Class"

TARGET_LABELS = ["legit", "fraudulent"]

# NUMERICAL_FEATURE_NAMES = [
#     "trip_seconds",
#     "trip_miles",
#     "euclidean",
# ]

# EMBEDDING_CATEGORICAL_FEATURES = {
#     "trip_month": 2,
#     "trip_day": 4,
#     "trip_hour": 3,
#     "pickup_grid": 3,
#     "dropoff_grid": 3,
#     "loc_cross": 10,
# }

# ONEHOT_CATEGORICAL_FEATURE_NAMES = ["payment_type", "trip_day_of_week"]


def transformed_name(key: str) -> str:
    """Generate the name of the transformed feature from original name."""
    return f"{key}_xf"


def original_name(key: str) -> str:
    """Generate the name of the original feature from transformed name."""
    return key.replace("_xf", "")


def vocabulary_name(key: str) -> str:
    """Generate the name of the vocabulary feature from original name."""
    return f"{key}_vocab"


def categorical_feature_names() -> list:
    return (
        list(EMBEDDING_CATEGORICAL_FEATURES.keys()) + ONEHOT_CATEGORICAL_FEATURE_NAMES
    )


def generate_explanation_config(transform_feature_spec):
    explanation_config = {
        "inputs": {},
        "outputs": {},
        "params": {"sampled_shapley_attribution": {"path_count": 10}},
    }

    for feature_name in transform_feature_spec:
        if feature_name != TARGET_FEATURE_NAME:
            explanation_config["inputs"][feature_name] = {
                "input_tensor_name": feature_name,
                "modality": "numeric",
            }
 
    explanation_config["outputs"] = {"scores": {"output_tensor_name": "scores"}}

    return explanation_config
