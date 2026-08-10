"""
run_utils.py -- RoadVision Nepal Phase 2: evidence-file scaffolding.

Creates a timestamped run folder and writes the reproducibility artifacts agreed
on for every training run:
    config.yaml       -- full hyperparameter set for this run
    pip_freeze.txt     -- exact installed package versions
    commit_hash.txt    -- git commit the run was executed at (+ dirty-tree warning)
    metrics.csv          -- one row per epoch, appended during training

Run folders are named by timestamp (runs/YYYY-MM-DD_HHMM/), per the naming
convention locked earlier -- zero-effort, always unique.
"""

import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml


def create_run_dir(base_dir: str = "runs") -> Path:
    """Creates and returns a new timestamped run directory: base_dir/YYYY-MM-DD_HHMM/"""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    run_dir = Path(base_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_config(run_dir: Path, config: dict) -> None:
    """Writes the full config dict used for this run to config.yaml."""
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def save_pip_freeze(run_dir: Path) -> None:
    """Writes `pip freeze` output to pip_freeze.txt for exact package-version reproducibility."""
    result = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
    with open(run_dir / "pip_freeze.txt", "w") as f:
        f.write(result.stdout)


def save_commit_hash(run_dir: Path) -> None:
    """
    Writes the current git commit hash to commit_hash.txt. If the working tree has
    uncommitted changes, that's flagged explicitly in the file -- a clean hash next
    to a dirty tree is misleading, since the recorded hash wouldn't actually match
    the code that ran.
    """
    try:
        commit_hash = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(status)
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit_hash = "UNKNOWN (git not available or this is not a git repo)"
        dirty = False

    with open(run_dir / "commit_hash.txt", "w") as f:
        f.write(f"{commit_hash}\n")
        if dirty:
            f.write("WARNING: working tree had uncommitted changes at run time -- "
                     "this hash may not exactly match the code that produced these results.\n")


def init_metrics_csv(run_dir: Path, fieldnames: list) -> Path:
    """Creates metrics.csv with a header row. Returns the path for later appending."""
    metrics_path = run_dir / "metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
    return metrics_path


def append_metrics_row(metrics_path: Path, row: dict) -> None:
    """Appends one epoch's results as a row to metrics.csv."""
    with open(metrics_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writerow(row)


def init_run(config: dict, base_dir: str = "runs", metrics_fields: list = None) -> dict:
    """
    Convenience wrapper: creates the run dir and writes config.yaml, pip_freeze.txt,
    commit_hash.txt, and an empty metrics.csv in one call.

    Returns {'run_dir': Path, 'metrics_path': Path}
    """
    run_dir = create_run_dir(base_dir)
    save_config(run_dir, config)
    save_pip_freeze(run_dir)
    save_commit_hash(run_dir)

    if metrics_fields is None:
        metrics_fields = ["epoch", "train_loss", "train_ce", "train_dice",
                           "val_loss", "val_ce", "val_dice",
                           "iou", "precision", "recall", "encoder_frozen"]
    metrics_path = init_metrics_csv(run_dir, metrics_fields)

    return {"run_dir": run_dir, "metrics_path": metrics_path}


if __name__ == "__main__":
    # Smoke test -- runs in a throwaway temp dir, not the real runs/ folder.
    import shutil
    import tempfile

    tmp = tempfile.mkdtemp()
    dummy_config = {"model": {"encoder": "resnet50"}, "train": {"lr": 1e-4, "batch_size": 8}}

    paths = init_run(dummy_config, base_dir=tmp)
    print(f"Run dir created: {paths['run_dir']}")

    assert (paths["run_dir"] / "config.yaml").exists()
    assert (paths["run_dir"] / "pip_freeze.txt").exists()
    assert (paths["run_dir"] / "commit_hash.txt").exists()
    assert (paths["run_dir"] / "metrics.csv").exists()
    print("All four evidence files created.")

    append_metrics_row(paths["metrics_path"], {
        "epoch": 0, "train_loss": 0.9, "train_ce": 0.5, "train_dice": 0.4,
        "val_loss": 0.8, "val_ce": 0.45, "val_dice": 0.35,
        "iou": 0.3, "precision": 0.5, "recall": 0.4, "encoder_frozen": True,
    })

    with open(paths["metrics_path"]) as f:
        content = f.read()
    print(f"\nmetrics.csv content:\n{content}")
    assert "0.9" in content

    print("run_utils.py smoke test passed.")
    shutil.rmtree(tmp)
