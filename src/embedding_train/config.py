from pathlib import Path

import hydra
from omegaconf import OmegaConf


CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def load_base_config():
    with hydra.initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        return hydra.compose(config_name="config")


def build_config_from_hyperparameters(hyperparameters):
    cfg = load_base_config()

    if hyperparameters is None:
        return cfg

    return OmegaConf.merge(cfg, OmegaConf.create(hyperparameters))
