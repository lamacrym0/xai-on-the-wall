NUM_EPOCHS = 50
LEARNING_RATE = 1e-3

VAL_SPLIT = 0.25
TEST_SPLIT = 0.2
RANDOM_SEED = None
SHUFFLE_DATASET = True

DEVICE = "cpu"


import random

import yaml
from pathlib import Path

class Config:
    """
    Wrapper around the YAML config file.
    Handles dataset loader creation and parameter access.
    """
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the Config object by loading the YAML configuration file.
        :param config_path: Path to the YAML configuration file (default: "config.yaml").
        :raises FileNotFoundError: If the specified config file does not exist.
        """
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_file, "r", encoding="utf-8") as f:
            self._cfg = yaml.safe_load(f)


    @property
    def seed(self) -> int:
        value = self._cfg.get("seed", 42)
        if isinstance(value, int):
            return value
        elif isinstance(value, str) and (value.lower() == "random" or value.lower() == "none"):
            return random.randint(0, 2 ** 32 - 1)
        else:
            return 42

    @property
    def ga_params(self) -> dict:
        return self._cfg.get("ga", {})

    def get(self, key: str, default=None):
        return self._cfg.get(key, default)

    def __getitem__(self, key):
        return self._cfg[key]
