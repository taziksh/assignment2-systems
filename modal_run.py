import tyro
import modal
from cs336_systems.config import BenchmarkConfig

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("uv")
        .add_local_dir("../assignment1-basics/", remote_path="/root/assignment1-basics", copy=True, ignore=[".venv", ".git", "data", "runs", "wandb", "__pycache__", "*.pyc", "tests", "*.pdf"])
        .add_local_file("pyproject.toml", "/root/assignment2-systems/pyproject.toml", copy=True)
        .add_local_file("uv.lock", "/root/assignment2-systems/uv.lock", copy=True)
        .run_commands(
            "uv pip install --system -e /root/assignment1-basics",
            "cd /root/assignment2-systems && uv pip install --system -r pyproject.toml"
        )
        .add_local_dir(".", remote_path="/root/assignment2-systems", copy=False, ignore=[".venv", ".git", "data", "runs", "wandb", "__pycache__", "*.pyc"])
        .add_local_python_source("cs336_systems")
)

app = modal.App("cs336-systems", image=image)
volume = modal.Volume.from_name("cs336-data", create_if_missing=True)

@app.function(gpu="H100", volumes={"/data": volume}, timeout=3600)
def run(size, mode, warmup_steps):
    from cs336_systems.benchmark import train
    from cs336_systems.config import BenchmarkConfig

    cfg = BenchmarkConfig(size=size, mode=mode, warmup_steps=warmup_steps)
    return train(cfg)

@app.local_entrypoint()
def main(*argv):
    cfg = tyro.cli(BenchmarkConfig, args=argv)
    import pandas as pd
    from cs336_systems.config import SIZES, MODES

    pairs = [(s, m) for s in SIZES for m in MODES]
    rows = list(run.starmap(pairs, return_exceptions=True, kwargs={"warmup_steps": cfg.warmup_steps}))
    rows = [r for r in rows if isinstance(r, dict)]
    df = pd.DataFrame(rows)
    df = df.pivot(index="size", columns="mode", values=["mean", "std"])
    df = df.reindex(list(SIZES))
    print(df.style.to_typst())

