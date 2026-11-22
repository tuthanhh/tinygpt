"""Resuming a run from a checkpoint.

    uv run pytest tests/test_resume.py
"""

import csv

import pytest
import torch

from tinygpt.checkpoint import latest_checkpoint
from tinygpt.train import train


@pytest.fixture
def run_kwargs(tmp_path):
    text = "the quick brown fox jumps over the lazy dog. " * 400
    (tmp_path / "train.txt").write_text(text)
    (tmp_path / "valid.txt").write_text(text[:4000])
    return dict(
        train_path=str(tmp_path / "train.txt"),
        val_path=str(tmp_path / "valid.txt"),
        train_max_bytes=None,
        cache_dir=None,
        tokenizer_spec="byte",
        context_length=16,
        d_model=32,
        num_layers=2,
        num_heads=2,
        d_ff=64,
        rope_theta=10000.0,
        batch_size=4,
        max_steps=20,
        max_lr=1e-3,
        min_lr=1e-4,
        warmup_iters=2,
        log_every=1,
        eval_every=5,
        eval_batches=2,
        ckpt_every=10,
        device="cpu",
        run_dir=str(tmp_path / "run"),
    )


def log_steps(run_dir) -> list[int]:
    with open(f"{run_dir}/train_log.csv", newline="") as f:
        return [int(row["step"]) for row in csv.DictReader(f)]


def read_rows(run_dir) -> dict[int, dict]:
    with open(f"{run_dir}/train_log.csv", newline="") as f:
        return {int(r["step"]): r for r in csv.DictReader(f)}


def test_latest_checkpoint_compares_steps_numerically(tmp_path):
    assert latest_checkpoint(tmp_path) is None
    for step in (900, 1200, 10000):
        (tmp_path / f"ckpt_{step}.pt").touch()
    (tmp_path / "ckpt_10000.pt.tmp").touch()  # an interrupted save is never picked
    assert latest_checkpoint(tmp_path).endswith("ckpt_10000.pt")


def test_resume_continues_the_run(run_kwargs):
    run_dir = run_kwargs["run_dir"]
    train(**run_kwargs)
    full_log = log_steps(run_dir)
    before = read_rows(run_dir)
    assert full_log == list(range(20)) + [20]  # steps 0..19, then the final eval row

    # As if the session died after ckpt_10: the log already has rows past it (10..20).
    out = train(**run_kwargs, resume_from=f"{run_dir}/ckpt_10.pt")
    resumed_log = log_steps(run_dir)
    assert resumed_log == full_log  # rows 10..19 were redone once, not appended twice

    final = torch.load(f"{run_dir}/ckpt_20.pt", map_location="cpu")
    assert final["iteration"] == 20
    assert out["tokens_seen"] == 20 * 4 * 16

    # The resumed segment started from the checkpoint's exact weights: val batches are fixed
    # (eval_seed), so the val loss at step 10 matches the uninterrupted run to the digit.
    after = read_rows(run_dir)
    assert after[10]["val_loss"] == before[10]["val_loss"]
    assert after[5] == before[5]  # rows before the checkpoint are kept untouched
    assert float(after[10]["elapsed_s"]) >= float(after[9]["elapsed_s"])  # time axis continues


def test_resume_rejects_a_different_model_shape(run_kwargs):
    train(**{**run_kwargs, "max_steps": 10})
    with pytest.raises(ValueError, match="num_layers"):
        train(**{**run_kwargs, "num_layers": 3}, resume_from=f"{run_kwargs['run_dir']}/ckpt_10.pt")
