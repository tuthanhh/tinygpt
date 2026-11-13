from tinygpt.train import train

# ---- data
TRAIN_PATH = "data/vi_wiki_noaccents/train.txt"
VAL_PATH = "data/vi_wiki_noaccents/valid.txt"
TRAIN_MAX_BYTES = 100 * 2**20  # first ~100MB, cut at a story boundary
CACHE_DIR = "data/cache"

# ---- model (vocab_size comes from the tokenizer)
CONTEXT_LENGTH = 256
D_MODEL = 256
NUM_LAYERS = 4
NUM_HEADS = 4
D_FF = 704
ROPE_THETA = 10_000.0

# ---- optimization
BATCH_SIZE = 32
MAX_STEPS = 3000
LR = 1e-1 # SGD: sweep this, it's the main knob

# ---- logging / eval
LOG_EVERY = 100
EVAL_EVERY = 500
EVAL_BATCHES = 20
RUN_DIR = "runs/vi-wiki_noaccents"
CKPT_EVERY = 500

# ---- reproducibility / debugging
SEED = 0
OVERFIT_ONE_BATCH = False  # gate: True -> train loss should head toward 0


if __name__ == "__main__":
    train(
        train_path=TRAIN_PATH,
        val_path=VAL_PATH,
        train_max_bytes=TRAIN_MAX_BYTES,
        cache_dir=CACHE_DIR,
        context_length=CONTEXT_LENGTH,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        rope_theta=ROPE_THETA,
        batch_size=BATCH_SIZE,
        max_steps=MAX_STEPS,
        lr=LR,
        log_every=LOG_EVERY,
        ckpt_every=CKPT_EVERY,
        eval_every=EVAL_EVERY,
        eval_batches=EVAL_BATCHES,
        run_dir=RUN_DIR,
        seed=SEED,
        overfit_one_batch=OVERFIT_ONE_BATCH,
    )
