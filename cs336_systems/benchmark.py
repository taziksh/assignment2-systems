import tyro
import torch
from einops import rearrange
import timeit
import statistics
from cs336_basics.transformer import TransformerLM
from cs336_basics.trainer import AdamWOptim, cross_entropy
from cs336_systems.config import BenchmarkConfig

def step(x, y, model, optim, mode, device):
    logits = model(x)
    logits = rearrange(logits, "b s v -> (b s) v")
    loss = cross_entropy(logits, y)

    if mode != "forward":
        optim.zero_grad()
        loss.backward()

    if mode == "full":
        optim.step()

    if device == "cuda":
        torch.cuda.synchronize()

def train(cfg):
    model = TransformerLM(
        cfg.model.vocab_size, cfg.model.context_length, cfg.model.d_model, cfg.model.num_layers, cfg.model.num_heads, cfg.model.d_ff, cfg.model.rope_theta
    ).to(cfg.device)

    optim = AdamWOptim(
        model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay, eps=cfg.optim.eps, betas=[cfg.optim.beta_1, cfg.optim.beta_2]
    )

    mode = cfg.mode
    device = cfg.device
    warmup_steps = cfg.warmup_steps
    measure_steps = cfg.measure_steps
    batch_size = cfg.batch_size
    context_length = cfg.model.context_length
    vocab_size = cfg.model.vocab_size

    x = torch.randint(low=0, high=vocab_size, size=(batch_size, context_length)).to(device)
    y = torch.randint(low=0, high=vocab_size, size=(batch_size, context_length)).to(device)
    y = rearrange(y, "b s -> (b s)")

    for _ in range(warmup_steps):
        step(x, y, model, optim, mode, device)


    times = []
    for _ in range(measure_steps):
        start_time = timeit.default_timer()
        step(x, y, model, optim, mode, device)
        elapsed_time = timeit.default_timer() - start_time
        times.append(elapsed_time)

    mean = statistics.mean(times)
    stdev = statistics.stdev(times)
    print(f"size={cfg.size} ran mode={mode.upper()} in {mean:.4f}s std={stdev}")
    return {"size": cfg.size, "mode": mode, "mean": mean, "std": stdev}

if __name__ == "__main__":
    cfg = tyro.cli(BenchmarkConfig)
    train(cfg)
