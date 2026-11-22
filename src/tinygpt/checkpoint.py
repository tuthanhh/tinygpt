import typing

from torch import nn
import torch
import os


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    config: dict | None = None,
) -> None:
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
        "config": config,
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    checkpoint = torch.load(src, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]


def load_config(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]) -> dict | None:
    """The config stored with a checkpoint, or None if it was saved without one."""
    return torch.load(src, map_location="cpu").get("config")


def latest_checkpoint(run_dir: str | os.PathLike) -> str | None:
    """Path of the ckpt_<step>.pt with the highest step in run_dir, or None if there is none.
    Compared by step number, not by name: ckpt_9000 < ckpt_12000."""
    from pathlib import Path

    steps = []
    for p in Path(run_dir).glob("ckpt_*.pt"):
        tail = p.stem.removeprefix("ckpt_")
        if tail.isdigit():
            steps.append((int(tail), p))
    return str(max(steps)[1]) if steps else None
