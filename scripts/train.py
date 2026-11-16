"""Train one run from a JSON config: one config file = one run = one folder.

    uv run python scripts/train.py configs/vi_noaccent_adamw.json
    uv run python scripts/train.py configs/vi_noaccent_adamw.json --run-dir runs/adamw_retry

Keys in the file are train()'s keyword arguments. JSON has no comments, so keys
starting with "_" are notes and are ignored. The run folder defaults to
runs/<config file name>; train() writes the resolved config.json there.
"""

import argparse
import json
from pathlib import Path

from tinygpt.train import train


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        config = json.load(f)
    config = {k: v for k, v in config.items() if not k.startswith("_")}
    if "run_dir" in config:
        raise SystemExit(f"{path}: put the run folder on the command line (--run-dir), not in the config")
    if "betas" in config:
        config["betas"] = tuple(config["betas"])  # JSON gives a list; AdamW expects a pair
    return config

def main() -> None:
    p = argparse.ArgumentParser(description="Train a tiny GPT from a JSON config.")
    p.add_argument("config", type=Path, help="JSON file with train() keyword arguments")
    p.add_argument("--run-dir", type=Path, default=None, help="default: runs/<config file name>")
    args = p.parse_args()

    config = load_config(args.config)
    run_dir = args.run_dir or Path("runs") / args.config.stem
    train(**config, run_dir=str(run_dir))


if __name__ == "__main__":
    main()
