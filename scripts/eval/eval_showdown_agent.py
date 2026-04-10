"""
Pokémon Showdown LLM agent — batch evaluation (raw log inference)
=================================================================
Evaluates a fine-tuned model that predicts the winner's next action (move/switch)
from raw Pokemon Showdown replay log prefixes.

Usage (from repository root):
  python3 scripts/eval/eval_showdown_agent.py --model_path /path/to/merged_or_checkpoint --samples 50

Metrics and sampling match `scripts/eval/showdown_agent_eval.py` (used by the ROCm tutorial notebook).
"""
import argparse
import os
import random
import sys

import torch
from datasets import load_dataset
from unsloth import FastLanguageModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

cwd = os.getcwd()
if cwd in sys.path and os.path.isdir(os.path.join(cwd, "unsloth-src")):
    sys.path = [p for p in sys.path if p != cwd]

from showdown_agent_eval import (  # noqa: E402
    build_test_samples,
    eval_showdown_agent_batch,
    print_metrics_summary,
    save_metrics_json,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to merged model or checkpoint directory",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=50,
        help="Number of real battle samples to evaluate",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=30,
        help="Max tokens to generate (outputs are short action lines)",
    )
    parser.add_argument(
        "--load_4bit",
        action="store_true",
        default=False,
        help="Load in 4bit quantization",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_json",
        type=str,
        default="",
        help="Where to write metrics JSON (default: ./eval_results_showdown_{samples}samples.json)",
    )
    parser.add_argument(
        "--no_use_cache",
        action="store_true",
        help="Pass use_cache=False to generate (often safer on ROCm + Qwen3)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"\n{'=' * 60}")
    print("Pokémon Showdown agent — evaluation")
    print(f"{'=' * 60}")
    print(f"  Model: {args.model_path}")
    print(f"  Samples: {args.samples}")

    if not os.path.isdir(args.model_path):
        print(f"\nError: Model path {args.model_path} does not exist!")
        return 1

    print("\nLoading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=4096,
        dtype=torch.bfloat16,
        load_in_4bit=args.load_4bit,
    )
    FastLanguageModel.for_inference(model)
    print("Model loaded.")

    print(f"Streaming dataset to collect {args.samples} evaluation samples...")
    dataset = load_dataset(
        "milkkarten/pokemon-showdown-replays-merged",
        split="train",
        streaming=True,
    )
    dataset = dataset.shuffle(seed=args.seed, buffer_size=10_000)
    test_samples = build_test_samples(dataset, args.samples, seed=args.seed)
    print(f"Prepared {len(test_samples)} test samples.")

    if not test_samples:
        print("No samples collected; aborting.")
        return 1

    def _progress(i, n, row):
        t_icon = "Y" if row["type_match"] else "N"
        e_icon = "Y" if row["exact_match"] else "N"
        print(
            f"[{i:2d}/{n}] R{row['rating']:.0f} {str(row['format'])[:8]:8s} | {row['elapsed_s']:.2f}s "
            f"type={t_icon} exact={e_icon}"
        )

    print(f"\n{'=' * 60}")
    print("Starting inference...")
    print(f"{'=' * 60}\n")

    metrics, _rows = eval_showdown_agent_batch(
        model,
        tokenizer,
        test_samples,
        max_new_tokens=args.max_tokens,
        use_cache=not args.no_use_cache,
        progress=_progress,
    )

    print_metrics_summary(metrics, title="Evaluation summary")

    out_path = args.output_json
    if not out_path:
        out_path = os.path.join(
            os.getcwd(), f"eval_results_showdown_{args.samples}samples.json"
        )
    save_metrics_json(
        out_path,
        metrics,
        extra={
            "model_path": args.model_path,
            "samples": args.samples,
            "seed": args.seed,
            "script": "eval_showdown_agent.py",
        },
    )
    print(f"\nResults saved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main() or 0)
