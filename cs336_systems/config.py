from cs336_basics.config import ModelConfig, OptimConfig
from dataclasses import dataclass, field
from typing import Literal, get_args

Mode = Literal["forward", "forward_backward", "full"]
MODES = get_args(Mode)

SIZES = {
    "small": {
        "d_model": 768,
        "d_ff": 3072,
        "num_layers": 12,
        "num_heads": 12
    }, 
    "medium": {
        "d_model": 1024,
        "d_ff": 4096,
        "num_layers": 24,
        "num_heads": 16
    }, 
    "large": {
        "d_model": 1280,
        "d_ff": 5120,
        "num_layers": 36,
        "num_heads": 20
    }, 
    "xl": {
        "d_model": 2560,
        "d_ff": 10240,
        "num_layers": 32, 
        "num_heads": 32
    }, 
    "10b": {
        "d_model": 4608,
        "d_ff": 12288,
        "num_layers": 50,
        "num_heads": 36
    }, 
}

@dataclass
class BenchmarkConfig:
    mode: Mode = "full"
    device: Literal["mps", "cuda", "cpu"] = "cuda"
    size: Literal[tuple(SIZES.keys())] = "small"
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    warmup_steps: int = 5
    measure_steps: int = 10
    batch_size: int = 4

    def __post_init__(self):
        for k, v in SIZES[self.size].items():
            setattr(self.model, k, v)


