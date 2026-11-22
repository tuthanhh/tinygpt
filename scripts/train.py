"""Train one run from a JSON config: one config file = one run = one folder.

    uv run python scripts/train.py configs/vi_noaccent_adamw.json
    uv run python scripts/train.py configs/vi_noaccent_adamw.json --run-dir runs/adamw_retry
    uv run python scripts/train.py configs/vi_noaccent_adamw.json --resume latest

Keys in the file are train()'s keyword arguments. JSON has no comments, so keys
starting with "_" are notes and are ignored. The run folder defaults to
runs/<config file name>; train() writes the resolved config.json there.
"""

import argparse
import json
from pathlib import Path

from tinygpt.checkpoint import latest_checkpoint
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
    p.add_argument(
        "--resume",
        default=None,
        help="a checkpoint to continue from, or 'latest' for the newest ckpt_*.pt in the run folder "
        "(starts fresh if it has none). The config must describe the same run.",
    )
    args = p.parse_args()

    config = load_config(args.config)
    run_dir = args.run_dir or Path("runs") / args.config.stem
    resume_from = args.resume
    if resume_from == "latest":
        resume_from = latest_checkpoint(run_dir)
        print(f"resume: {resume_from or f'no checkpoint in {run_dir}, starting fresh'}")
    elif resume_from is not None and not Path(resume_from).is_file():
        raise SystemExit(f"--resume {resume_from}: no such checkpoint")
    train(**config, run_dir=str(run_dir), resume_from=resume_from)


if __name__ == "__main__":
    main()
