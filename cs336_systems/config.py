from cs336_basics.config import ModelConfig, OptimConfig
from dataclasses import dataclass, field
from typing import Literal

@dataclass
class BenchmarkConfig:
    mode: Literal["forward", "forward_backward", "forward_backward_optim"] = "forward_backward_optim"
    device: Literal["mps", "cuda", "cpu"] = "cuda"
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    warmup_steps: int = 5
    measure_steps: int = 10
    batch_size: int = 4
    num_data: int = 256
