import csv
import math
import time
from pathlib import Path

import numpy as np
import torch

from tinygpt.checkpoint import save_checkpoint
from tinygpt.data import get_batch, load_tokens
from tinygpt.losses import cross_entropy
from tinygpt.model import TransformerLM
from tinygpt.optim import SGD
from tinygpt.tokenizer.byte import ByteTokenizer


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
    lr: float,
    # logging / eval
    log_every: int = 10,
    eval_every: int = 200,
    eval_batches: int = 20,
    ckpt_every: int = 500,
    run_dir: str = "runs/model",
    # reproducibility / debugging
    seed: int = 0,
    eval_seed: int = 1234,
    overfit_one_batch: bool = False,
    device: str | torch.device | None = None,
) -> dict:
    # ---------------------------------------------------------------- 1. setup

    device = torch.device(device) if device is not None else pick_device()
    tokenizer = ByteTokenizer()

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
    ckpt_config = {**model_config, "tokenizer": tokenizer.name}

    model = TransformerLM(**model_config, device=device)
    model.train()
    optimizer = SGD(model.parameters(), lr=lr)

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"device={device}  params={n_params:,}  train_tokens={len(train_tokens):,}  "
        f"val_tokens={len(val_tokens):,}  expected step-0 loss≈ln({tokenizer.vocab_size})"
        f"={math.log(tokenizer.vocab_size):.3f}"
    )

    fixed_batch = get_batch(train_tokens, batch_size, context_length, device) if overfit_one_batch else None
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    log_file = open(f"{run_dir}/train_log.csv", "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow(["step", "train_loss", "val_loss", "lr", "elapsed_s", "tokens_seen"])

    tokens_per_step = batch_size * context_length
    start = time.perf_counter()
    train_loss = float("nan")
    val_loss = float("nan")

    # ---------------------------------------------------------------- 2. loop
    try:
        for step in range(max_steps):
            x, y = (
                fixed_batch if fixed_batch is not None else get_batch(train_tokens, batch_size, context_length, device)
            )

            logits = model(x)  # (B, L, V)
            loss = cross_entropy(logits, y)

            # ---- 3/4. logging and validation, on the loss BEFORE this step's
            # update, so step 0 shows the untrained model (gate: ≈ ln V).
            do_log = step % log_every == 0
            do_eval = step % eval_every == 0
            do_save = step % ckpt_every == 0
            if do_log or do_eval:
                train_loss = loss.item()  # syncs with the GPU: only on log steps
                val_str = ""
                if do_eval:
                    val_loss = evaluate(model, val_batch_list)
                    val_str = f"{val_loss:.4f}"
                elapsed = time.perf_counter() - start
                lr_now = optimizer.param_groups[0]["lr"]
                seen = step * tokens_per_step
                writer.writerow([step, f"{train_loss:.4f}", val_str, lr_now, f"{elapsed:.1f}", seen])
                log_file.flush()
                print(
                    f"step {step:6d}  train {train_loss:.4f}"
                    + (f"  val {val_str}" if val_str else "")
                    + f"  {elapsed:7.1f}s"
                )
            if do_save:
                save_checkpoint(model, optimizer, step, Path(run_dir) / f"ckpt_{step}.pt", ckpt_config)
            optimizer.zero_grad()  # gradients accumulate: clear first
            loss.backward()
            optimizer.step()

        # ------------------------------------------------------------ 5. end
        val_loss = evaluate(model, val_batch_list)
        elapsed = time.perf_counter() - start
        seen = max_steps * tokens_per_step
        writer.writerow([max_steps, "", f"{val_loss:.4f}", optimizer.param_groups[0]["lr"], f"{elapsed:.1f}", seen])
    finally:
        log_file.close()  # closed even if training crashes

    summary = {
        "steps": max_steps,
        "elapsed_s": elapsed,
        "tokens_seen": seen,
        "last_logged_train_loss": train_loss,
        "final_val_loss": val_loss,
        "params": n_params,
    }
    save_checkpoint(model, optimizer, max_steps, Path(run_dir) / f"ckpt_{max_steps}.pt", ckpt_config)
    print(
        f"done: {max_steps} steps in {elapsed:.1f}s, {seen:,} tokens, "
        f"last train {train_loss:.4f}, final val {val_loss:.4f}"
    )
    return summary
