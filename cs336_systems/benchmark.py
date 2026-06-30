import tyro
import torch
from jaxtyping import Float, Bool
import math
from einops import rearrange, einsum
import timeit
import statistics
from cs336_basics.transformer import TransformerLM, softmax
from cs336_basics.trainer import AdamWOptim, cross_entropy
import cs336_basics.transformer
import torch.cuda.nvtx as nvtx
from cs336_systems.config import BenchmarkConfig

def annotated_scaled_dot_product_attention(
    Q: Float[torch.Tensor, " ... queries d_k"],
    K: Float[torch.Tensor, " ... keys    d_k"],
    V: Float[torch.Tensor, " ... keys    d_v"],
    mask: Bool[torch.Tensor, " ... queries keys"] | None = None,
) -> Float[torch.Tensor, " ... queries d_v"]:
    """Scaled dot-product attention.

    This function implements Eq. 1 of the Transformer paper.

    Args:
        Q: Tensor of queries, may have any number of leading dimensions.
        K: Tensor of keys, sharing leading dimensions with Q.
        V: Tensor of values, sharding leading dimensions with Q and K.
        mask: An (optional) mask of shape (..., seq_len, seq_len).
            Attention scores for positions with a mask value of `False` should
            be masked out, i.e., not affect the softmaxed attention probabilities.

    Returns:
        torch.FloatTensor of shape (..., seq_len, value_dimension)
        with the output of running your scaled dot product attention
        implementation with the provided key, query, and value tensors.
    """

    d_k = K.shape[-1]
    with nvtx.range("computing attention scores"):
        attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)

    if mask is not None:
        attention_scores = torch.where(mask, attention_scores, float("-inf"))

    with nvtx.range("computing softmax"):
        attention_weights = softmax(attention_scores, dim=-1)  # Softmax over the key dimension

    with nvtx.range("final matmul"):
        out = einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")
    return out

cs336_basics.transformer.scaled_dot_product_attention = annotated_scaled_dot_product_attention

def step(x, y, model, optim, mode, device):
    with nvtx.range("forward pass"):
        logits = model(x)
        logits = rearrange(logits, "b s v -> (b s) v")
        loss = cross_entropy(logits, y)

    if mode != "forward":
        with nvtx.range("backward pass"):
            optim.zero_grad()
            loss.backward()

    if mode == "full":
        with nvtx.range("optimizer"):
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

    with nvtx.range("warmup"):
        for _ in range(warmup_steps):
            step(x, y, model, optim, mode, device)

    if cfg.profile_memory:
        # Start recording memory history
        torch.cuda.memory._record_memory_history(max_entries=1000000)

    times = []
    with nvtx.range("measured"):
        for _ in range(measure_steps):
            start_time = timeit.default_timer()
            step(x, y, model, optim, mode, device)
            elapsed_time = timeit.default_timer() - start_time
            times.append(elapsed_time)

    if cfg.profile_memory:
        # Save pickle file to load with PyTorch's online tool
        torch.cuda.memory._dump_snapshot(f"profiles/memory_{cfg.size}_ctx{context_length}_{mode}.pickle")

        # Stop recording memory history
        torch.cuda.memory._record_memory_history(enabled=None)

    mean = statistics.mean(times)
    stdev = statistics.stdev(times) if len(times) > 1 else math.nan
    print(f"size={cfg.size} ran mode={mode.upper()} in {mean:.4f}s std={stdev}")
    return {"size": cfg.size, "mode": mode, "mean": mean, "std": stdev}

if __name__ == "__main__":
    cfg = tyro.cli(BenchmarkConfig)
    train(cfg)
