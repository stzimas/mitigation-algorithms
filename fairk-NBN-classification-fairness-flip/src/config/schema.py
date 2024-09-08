from pydantic.dataclasses import dataclass
from pydantic import field_validator
from typing import List, Optional, Union
import hydra
from hydra.core.config_store import ConfigStore


@dataclass
class BasicConfig:
    neighbors: int

    distance: str
    split_data: dict
    split_percent: float
    feature_selection: bool
    exclude_sensitive_attribute: bool
    focus_metric: str

@dataclass
class ClassAttribute:
    name: str
    values: List[str]


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


# Register the configuration schema with Hydra
cs = ConfigStore.instance()
cs.store(name="config_schema", node=Config)
