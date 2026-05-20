from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ProjectConfig:
    target: str
    numerical_features: list[str]
    categorical_features: list[str]
    ignore_features: list[str] = field(default_factory=list)
    problem_type: Literal["classification", "regression"] = "classification"
    random_state: int = 1234
    test_size: float = 0.2
    cv_folds: int = 5
