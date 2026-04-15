from pathlib import Path

import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf


CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def load_base_config():
    with hydra.initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        return hydra.compose(config_name="config")


def build_config_from_hyperparameters(hyperparameters):
    if hyperparameters is None:
        return load_base_config()

    hparams_cfg = OmegaConf.create(hyperparameters)

    # Hparams are persisted with resolve=True, so they already contain every
    # field a module needs. When we're inside an active @hydra.main run (e.g.
    # loading a teacher during student training), re-initializing Hydra to
    # compose the base config would raise, so skip the merge in that case.
    if GlobalHydra.instance().is_initialized():
        return hparams_cfg

    try:
        return OmegaConf.merge(load_base_config(), hparams_cfg)
    except Exception:
        # Checkpoint hyper_parameters may contain keys that no longer exist in
        # the current base config (or vice-versa). Fall back to the resolved
        # checkpoint config which is self-contained.
        return hparams_cfg
