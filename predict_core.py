"""Shared prediction pipeline for the Vercel API and local tests."""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

import features as feat

DISPLAY_THRESHOLD = 0.5
SHORT_STROKE_MSG = "Need a longer continuous trace before a stable score is possible."


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _score_tree(tree: dict, features: list[float]) -> float:
    node_index = 0
    while True:
        left = tree["left_children"][node_index]
        right = tree["right_children"][node_index]
        if left == -1 and right == -1:
            return float(tree["base_weights"][node_index])

        feature_index = int(tree["split_indices"][node_index])
        threshold = float(tree["split_conditions"][node_index])
        default_left = bool(tree["default_left"][node_index])
        feature_value = features[feature_index]

        if feature_value is None:
            node_index = left if default_left else right
        elif feature_value < threshold:
            node_index = left
        else:
            node_index = right

        if node_index < 0:
            return 0.0


def _load_artifacts(artifact_dir: str):
    root = Path(artifact_dir)
    scaler = json.loads((root / "scaler.json").read_text(encoding="utf-8"))
    feature_cols = json.loads((root / "feature_columns.json").read_text(encoding="utf-8"))
    model_data = json.loads((root / "xgb_model.json").read_text(encoding="utf-8"))
    trees = model_data["learner"]["gradient_booster"]["model"]["trees"]

    base_score = model_data["learner"]["learner_model_param"].get("base_score", "[0.5]")
    base_score = base_score.strip()
    if base_score.startswith("[") and base_score.endswith("]"):
        base_score = base_score[1:-1]
    base_score = float(base_score)
    base_score_raw = 0.0
    if 0.0 < base_score < 1.0:
        base_score_raw = math.log(base_score / (1.0 - base_score))

    return trees, scaler, feature_cols, base_score_raw


def build_feature_row(raw_points: list) -> dict | None:
    if not raw_points:
        return None

    rows = [
        {
            "X": point["x"],
            "Y": point["y"],
            "Timestamp": point["t"],
            "Pressure": point["pressure"],
            "stroke": point["stroke"],
        }
        for point in raw_points
    ]

    raw_kinematics = feat.compute_point_kinematics(rows, ["stroke"])
    if len(raw_kinematics) < feat.MIN_KINEMATIC_ROWS:
        return None

    kinematics = [{"_session": 0, **entry} for entry in raw_kinematics]
    table = feat.aggregate_feature_table(kinematics, ["_session"], feat.AGG_FUNCS)
    if not table:
        return None

    output = table[0].copy()
    output.pop("_session", None)
    return output


@lru_cache(maxsize=4)
def _load_model_data(artifact_dir: str):
    return _load_artifacts(artifact_dir)


def predict_from_points(raw_points: list, artifact_dir: Path) -> dict:
    feature_row = build_feature_row(raw_points)
    if feature_row is None:
        raise ValueError(SHORT_STROKE_MSG)

    trees, scaler, feature_cols, base_score_raw = _load_model_data(str(artifact_dir.resolve()))
    feature_vector = []
    for index, col in enumerate(feature_cols):
        value = float(feature_row[col])
        if scaler.get("with_mean", False):
            value -= float(scaler["mean"][index])
        if scaler.get("with_std", False):
            divisor = float(scaler["scale"][index])
            value /= divisor if divisor != 0.0 else 1.0
        feature_vector.append(value)

    raw_score = base_score_raw
    for tree in trees:
        raw_score += _score_tree(tree, feature_vector)

    proba = _sigmoid(raw_score)
    flagged = proba >= DISPLAY_THRESHOLD

    return {
        "probability": proba,
        "flagged": flagged,
        "threshold": DISPLAY_THRESHOLD,
        "similarity": "parkinson" if flagged else "control",
        "features": {col: float(feature_row[col]) for col in feature_cols},
    }
