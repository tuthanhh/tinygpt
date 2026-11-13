import argparse
import codecs
from dataclasses import dataclass

try:
    import readline  # noqa: F401  -- arrow keys + history for input() on Linux/macOS
except ImportError:
    pass

import torch

# adjust these imports to your package layout
from tinygpt.generate import generate
from tinygpt.model import TransformerLM
from tinygpt.tokenizer.byte import ByteTokenizer

TOKENIZERS = {cls.name: cls for cls in (ByteTokenizer,)}

HELP = """\
Type a prompt and press Enter to continue it. Empty line = a fresh story.
Commands:
  :temp <float>     temperature (0 = greedy)
  :top_p <float>    nucleus threshold in (0, 1]
  :max <int>        max new tokens
  :seed <int|none>  fixed seed -> same prompt gives same output
  :show             current settings
  :help             this text
  :quit             exit (or Ctrl-D)
Ctrl-C while generating stops that generation and returns to the prompt."""


@dataclass
class Settings:
    temperature: float
    top_p: float
    max_new_tokens: int
    seed: int | None

    def __str__(self) -> str:
        return (f"temp={self.temperature}  top_p={self.top_p}  "
                f"max={self.max_new_tokens}  seed={self.seed}")


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(ckpt_path: str, device: str) -> tuple[TransformerLM, dict, str]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if ckpt.get("config") is None:
        raise SystemExit(f"{ckpt_path} has no config: it was saved before configs were stored.")
    config = dict(ckpt["config"])            # copy, so pop doesn't touch the loaded dict
    tokenizer_name = config.pop("tokenizer")
    model = TransformerLM(**config, device=torch.device(device))
    model.load_state_dict(ckpt["model"])     # no optimizer needed for generation
    model.eval()
    return model, config, tokenizer_name


def handle_command(line: str, s: Settings) -> bool:
    """Apply a ':command' to the settings. Returns False when the REPL should exit."""
    cmd, _, arg = line[1:].strip().partition(" ")
    arg = arg.strip()
    try:
        if cmd in ("quit", "q", "exit"):
            return False
        if cmd == "help":
            print(HELP)
            return True
        if cmd == "show":
            print(s)
            return True

        if cmd == "temp":
            value = float(arg)
            if value < 0:
                raise ValueError("must be >= 0")
            s.temperature = value
        elif cmd == "top_p":
            value = float(arg)
            if not 0 < value <= 1:
                raise ValueError("must be in (0, 1]")
            s.top_p = value
        elif cmd == "max":
            value = int(arg)
            if value <= 0:
                raise ValueError("must be > 0")
            s.max_new_tokens = value
        elif cmd == "seed":
            s.seed = None if arg.lower() in ("", "none") else int(arg)
        else:
            print(f"unknown command :{cmd}  (:help for the list)")
            return True
        print(s)                              # confirm the change
    except ValueError as e:
        print(f"bad value for :{cmd}: {e}")
    return True


def run_once(model, tokenizer, config: dict, prompt: str, s: Settings) -> None:
    # Fresh generator per prompt: with a fixed seed, the same prompt reproduces
    # the same output no matter what was generated before it in this session.
    generator = torch.Generator().manual_seed(s.seed) if s.seed is not None else None
    # Byte tokens can split a character; this holds partial UTF-8 until it completes.
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    print(prompt, end="", flush=True)
    try:
        for token_id in generate(
            model,
            tokenizer.encode(prompt),
            max_new_tokens=s.max_new_tokens,
            temperature=s.temperature,
            top_p=s.top_p,
            eot_id=tokenizer.eot_id,
            context_length=config["context_length"],
            generator=generator,
        ):
            print(decoder.decode(bytes([token_id])), end="", flush=True)
    except KeyboardInterrupt:
        print(" [stopped]", end="")
    print(decoder.decode(b"", final=True))   # flush any partial character, then newline


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive sampling from a TinyGPT checkpoint.")
    parser.add_argument("--ckpt", required=True, help="path to a checkpoint .pt file")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--seed", type=int, default=None, help="same seed -> same output")
    parser.add_argument("--device", default=None)
    parser.add_argument("--prompt", default=None, help="one-shot: generate once and exit")
    args = parser.parse_args()

    device = args.device or pick_device()
    model, config, tokenizer_name = load_model(args.ckpt, device)
    if tokenizer_name not in TOKENIZERS:
        raise SystemExit(f"checkpoint uses tokenizer {tokenizer_name!r}; known: {sorted(TOKENIZERS)}")
    tokenizer = TOKENIZERS[tokenizer_name]()
    s = Settings(args.temperature, args.top_p, args.max_new_tokens, args.seed)

    if args.prompt is not None:              # one-shot mode: scripts, reproducibility checks
        run_once(model, tokenizer, config, args.prompt, s)
        return

    n_params = sum(p.numel() for p in model.parameters())
    print(f"loaded {args.ckpt}  ({n_params:,} params, {device}, tokenizer={tokenizer_name})")
    print(HELP)
    print(s)
    while True:
        try:
            line = input("\n>>> ")
        except (EOFError, KeyboardInterrupt):  # Ctrl-D, or Ctrl-C at the prompt
            print()
            break
        if line.startswith(":"):
            if not handle_command(line, s):
                break
            continue
        run_once(model, tokenizer, config, line, s)


if __name__ == "__main__":
    main()
