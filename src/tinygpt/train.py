import csv
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from tinygpt.checkpoint import save_checkpoint
from tinygpt.data import get_batch, load_tokens
from tinygpt.losses import cross_entropy
from tinygpt.model import TransformerLM
from tinygpt.optim import SGD, AdamW, gradient_clipping, learning_rate_schedule
from tinygpt.tokenizer import load_tokenizer


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model: torch.nn.Module, batches: list[tuple[torch.Tensor, torch.Tensor]]) -> float:
    """Mean loss over a FIXED list of batches, so evaluations are comparable."""
    model.eval()
    total = 0.0
    for x, y in batches:
        total += cross_entropy(model(x), y).item()
    model.train()
    return total / len(batches)


def train(
    *,
    # data
    train_path: str,
    val_path: str,
    train_max_bytes: int | None,
    cache_dir: str | None = "data/cache",
    tokenizer_spec: str = "byte",  # "byte" or a folder from scripts/train_bpe.py
    # model
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    # optimization
    batch_size: int,
    max_steps: int,
    optimizer_name: str = "adamw",  # "adamw" or "sgd"
    max_lr: float = 3e-3,
    min_lr: float = 3e-4,
    warmup_iters: int = 200,
    cosine_cycle_iters: int | None = None,  # None -> max_steps: reach min_lr at the end
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.1,
    max_grad_norm: float = 1.0,
    # logging / eval / checkpoints
    log_every: int = 10,
    eval_every: int = 200,
    eval_batches: int = 20,
    ckpt_every: int = 500,
    run_dir: str = "runs/model",
    resume_from: str | None = None,  # a ckpt_<step>.pt of this same run: continue at that step
    # reproducibility / debugging
    seed: int = 0,
    eval_seed: int = 1234,
    overfit_one_batch: bool = False,
    device: str | torch.device | None = None,
) -> dict:
    # ---------------------------------------------------------------- 1. setup
    device = torch.device(device) if device is not None else pick_device()
    if cosine_cycle_iters is None:
        cosine_cycle_iters = max_steps
    assert cosine_cycle_iters > warmup_iters, "cosine phase needs at least one step"

    tokenizer = load_tokenizer(tokenizer_spec)
    train_tokens = load_tokens(train_path, tokenizer, max_bytes=train_max_bytes, cache_dir=cache_dir)
    val_tokens = load_tokens(val_path, tokenizer, max_bytes=None, cache_dir=cache_dir)

    # Fixed validation batches: drawn once, with their own seed, BEFORE the
    # training seed is set, so they don't depend on (or disturb) training's RNG.
    np.random.seed(eval_seed)
    val_batch_list = [get_batch(val_tokens, batch_size, context_length, device) for _ in range(eval_batches)]

    # Training seed: model init (torch) and batch sampling (numpy).
    torch.manual_seed(seed)
    np.random.seed(seed)

    model_config = {
        "vocab_size": tokenizer.vocab_size,
        "context_length": context_length,
        "d_model": d_model,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "d_ff": d_ff,
        "rope_theta": rope_theta,
    }
    model = TransformerLM(**model_config, device=device)
    model.train()

    # The lr passed here is only the starting value; the schedule overwrites it every step.
    if optimizer_name == "adamw":
        optimizer = AdamW(model.parameters(), lr=max_lr, betas=betas, eps=eps, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        optimizer = SGD(model.parameters(), lr=max_lr)
    else:
        raise ValueError(f"unknown optimizer {optimizer_name!r} (use 'adamw' or 'sgd')")

    def lr_at(step: int) -> float:
        return learning_rate_schedule(step, max_lr, min_lr, warmup_iters, cosine_cycle_iters)

    # Resume: weights + AdamW moments + step. The lr schedule is a function of the step, so it
    # continues by itself. ckpt_S is saved BEFORE step S's update: step S is the next to run.
    start_step = 0
    if resume_from is not None:
        ckpt = torch.load(resume_from, map_location="cpu")
        saved = ckpt.get("config") or {}
        mismatch = {k: (saved[k], v) for k, v in model_config.items() if k in saved and saved[k] != v}
        if mismatch:
            raise ValueError(
                f"{resume_from} has a different model shape (checkpoint, config): {mismatch}. "
                "Resume with the checkpoint's own settings."
            )
        saved_optim = saved.get("run", {}).get("optim", {})
        current_optim = {"max_lr": max_lr, "min_lr": min_lr, "warmup_iters": warmup_iters,
                         "cosine_cycle_iters": cosine_cycle_iters, "max_steps": max_steps, "batch_size": batch_size}
        changed = {k: (saved_optim[k], v) for k, v in current_optim.items() if k in saved_optim and saved_optim[k] != v}
        if changed:
            print(f"!! resuming with changed settings (checkpoint, now): {changed}")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])  # moves the moments to the params' device
        start_step = ckpt["iteration"]
        del ckpt
        # A different batch stream from the first segment's: seeding with `seed` again would
        # replay the batches of steps 0, 1, 2, ... instead of drawing fresh ones.
        np.random.seed(seed + start_step)
        print(f"resumed from {resume_from} at step {start_step}")

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"device={device}  params={n_params:,}  optimizer={optimizer_name}  "
        f"train_tokens={len(train_tokens):,}  val_tokens={len(val_tokens):,}  "
        f"expected step-0 loss≈ln({tokenizer.vocab_size})={math.log(tokenizer.vocab_size):.3f}"
    )

    # Everything needed to tell runs apart (and rerun one), in one dict. Plain
    # JSON types only, so it survives both json.dump and torch.load(weights_only=True).
    run_config = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "params": n_params,
        "data": {
            "train_path": train_path,
            "val_path": val_path,
            "tokenizer": tokenizer.name,
            "train_max_bytes": train_max_bytes,
            "train_tokens": len(train_tokens),
            "val_tokens": len(val_tokens),
        },
        "optim": {
            "optimizer": optimizer_name,
            "max_lr": max_lr,
            "min_lr": min_lr,
            "warmup_iters": warmup_iters,
            "cosine_cycle_iters": cosine_cycle_iters,  # resolved value, not None
            "betas": list(betas),
            "eps": eps,
            "weight_decay": weight_decay,
            "max_grad_norm": max_grad_norm,
            "batch_size": batch_size,
            "max_steps": max_steps,
        },
        "seed": seed,
        "eval_seed": eval_seed,
        "eval_batches": eval_batches,
        "overfit_one_batch": overfit_one_batch,
        "resumed_from": str(resume_from) if resume_from is not None else None,
        "resumed_at_step": start_step,
    }
    # Model keys stay top-level so the CLI can still build TransformerLM(**config)
    # after popping "tokenizer" and "run".
    # "tokenizer" holds the spec, not the name: it is what load_tokenizer needs to rebuild it.
    ckpt_config = {**model_config, "tokenizer": tokenizer_spec, "run": run_config}

    fixed_batch = get_batch(train_tokens, batch_size, context_length, device) if overfit_one_batch else None
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    with open(run_path / "config.json", "w", encoding="utf-8") as f:
        json.dump(ckpt_config, f, indent=2)  # same dict as every checkpoint carries
    header = ["step", "train_loss", "val_loss", "lr", "grad_norm", "elapsed_s", "tokens_seen"]
    log_path = run_path / "train_log.csv"
    # On resume, keep the log of the steps before start_step: rows from start_step on (up to
    # wherever the first segment died) are about to be redone, and would appear twice.
    kept_rows: list[list[str]] = []
    if start_step > 0 and log_path.exists():
        with open(log_path, newline="") as f:
            kept_rows = [row for row in csv.reader(f) if row and row[0].isdigit() and int(row[0]) < start_step]
    elapsed_offset = float(kept_rows[-1][5]) if kept_rows else 0.0  # time axis continues
    log_file = open(log_path, "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow(header)
    writer.writerows(kept_rows)
    log_file.flush()

    tokens_per_step = batch_size * context_length
    start = time.perf_counter() - elapsed_offset
    train_loss = float("nan")
    val_loss = float("nan")
    elapsed = elapsed_offset
    seen = start_step * tokens_per_step

    # ---------------------------------------------------------------- 2. loop
    # Everything logged for `step` describes the weights after `step` updates:
    # val loss, train loss (forward pass before this step's update), and the lr
    # and grad norm used for this step's update.
    def save(step: int) -> None:
        # Write a temp file, then rename: a session killed mid-save (a ~2GB file to Drive) leaves
        # the previous checkpoints intact instead of a truncated ckpt that resume would pick.
        final = run_path / f"ckpt_{step}.pt"
        tmp = final.with_suffix(".pt.tmp")
        save_checkpoint(model, optimizer, step, tmp, ckpt_config)
        os.replace(tmp, final)

    try:
        for step in range(start_step, max_steps):
            # lr for THIS step, set before anything reads it, on every param group.
            lr_now = lr_at(step)
            for group in optimizer.param_groups:
                group["lr"] = lr_now

            do_log = step % log_every == 0
            do_eval = step % eval_every == 0
            # Not at step 0 (untrained) nor at start_step (that checkpoint is where we resumed from).
            do_save = step > start_step and step % ckpt_every == 0

            if do_eval:
                val_loss = evaluate(model, val_batch_list)
            if do_save:
                save(step)

            x, y = (
                fixed_batch if fixed_batch is not None else get_batch(train_tokens, batch_size, context_length, device)
            )
            logits = model(x)  # (B, L, V)
            loss = cross_entropy(logits, y)

            optimizer.zero_grad()  # gradients accumulate: clear first
            loss.backward()
            grad_norm = gradient_clipping(model.parameters(), max_grad_norm)  # norm BEFORE clipping
            optimizer.step()

            # ---- 3/4. logging. .item() syncs with the GPU, so only on log steps.
            if do_log or do_eval:
                train_loss = loss.item()  # still the pre-update loss: step 0 ≈ untrained
                gn = float(grad_norm)
                val_str = f"{val_loss:.4f}" if do_eval else ""
                elapsed = time.perf_counter() - start
                seen = step * tokens_per_step
                writer.writerow(
                    [step, f"{train_loss:.4f}", val_str, f"{lr_now:.3e}", f"{gn:.3f}", f"{elapsed:.1f}", seen]
                )
                log_file.flush()
                print(
                    f"step {step:6d}  train {train_loss:.4f}"
                    + (f"  val {val_str}" if val_str else "")
                    + f"  lr {lr_now:.2e}  gnorm {gn:6.3f}  {elapsed:7.1f}s"
                )

        # ------------------------------------------------------------ 5. end
        val_loss = evaluate(model, val_batch_list)
        elapsed = time.perf_counter() - start
        seen = max_steps * tokens_per_step
        writer.writerow([max_steps, "", f"{val_loss:.4f}", "", "", f"{elapsed:.1f}", seen])
        save(max_steps)
    finally:
        log_file.close()  # closed even if training crashes

    print(
        f"done: {max_steps - start_step} steps (to step {max_steps}) in {elapsed:.1f}s, {seen:,} tokens, "
        f"last train {train_loss:.4f}, final val {val_loss:.4f}"
    )
    return {
        "steps": max_steps,
        "elapsed_s": elapsed,
        "tokens_seen": seen,
        "last_logged_train_loss": train_loss,
        "final_val_loss": val_loss,
        "params": n_params,
    }
