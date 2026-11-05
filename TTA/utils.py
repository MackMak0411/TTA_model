from __future__ import annotations

import yaml

# Read yaml file
def parse_yaml(config_yaml: str) -> dict:
    r"""Parse yaml file."""
    
    with open(config_yaml, "r") as fr:
        return yaml.load(fr, Loader=yaml.FullLoader)

# Changing the Learning Rate before the warm up steps, and keep it constant after the warm up steps
class LinearWarmUp:
    r"""Linear learning rate warm up scheduler.
    """
    def __init__(self, warm_up_steps: int) -> None:
        self.warm_up_steps = warm_up_steps

    def __call__(self, step: int) -> float:
        if step <= self.warm_up_steps:
            return step / self.warm_up_steps
        else:
            return 1.

# Truncate a list to a fixed length
def pad_or_truncate(x: list, length: int, pad_value) -> list:
    
    if len(x) >= length:
        return x[: length]
    
    else:
        return x + [pad_value] * (length - len(x))