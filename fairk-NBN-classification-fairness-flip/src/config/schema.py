from pydantic.dataclasses import dataclass
from pydantic import field_validator
from typing import List, Optional, Union
import hydra
from hydra.core.config_store import ConfigStore

@dataclass
class SplitData:
    val_data:bool
    resampling_train_set:bool
@dataclass
class Condition:
    affirmative_action:bool
    sensitive_catches_dominant:bool
    difference_percentage: int
@dataclass
class Weight:
    include_dominant_attribute:bool
    second_weight: bool

@dataclass
class BasicConfig:
    neighbors: int
    distance: str
    split_data: SplitData
    split_percent: float
    feature_selection: bool
    exclude_sensitive_attribute: bool
    focus_metric: str
    condition: Condition
    weight: Weight

@dataclass
class ClassAttribute:
    name: str
    positive_value: List[str]


@dataclass
class SensitiveAttribute:
    name: str
    protected: str




@dataclass
class DataConfig:
    load_from: str
    sensitive_attribute: SensitiveAttribute
    class_attribute: ClassAttribute
    results_path : str
    plot_path: str


@dataclass
class Config:
    defaults: Optional[list]
    data: DataConfig
    basic: BasicConfig
    experiment_name: str
    csv_to_word: bool


# Register the configuration schema with Hydra
cs = ConfigStore.instance()
cs.store(name="config_schema", node=Config)
