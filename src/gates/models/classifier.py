from __future__ import annotations

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeClassifier


def make_classifier(seed: int, kind: str = "logistic") -> Pipeline:
    if kind == "logistic":
        model = LogisticRegression(
            C=0.25,
            class_weight="balanced",
            max_iter=2_000,
            random_state=seed,
        )
    elif kind == "decision_tree":
        model = DecisionTreeClassifier(
            max_depth=4,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=seed,
        )
    else:
        raise ValueError(f"unknown classifier kind: {kind}")
    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", RobustScaler()),
            ("model", model),
        ]
    )
