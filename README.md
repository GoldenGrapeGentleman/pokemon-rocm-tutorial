# Pokémon LLM Agent with Unsloth (ROCm tutorial)

End-to-end **Jupyter** walkthrough for fine-tuning **`Qwen/Qwen3-4B`** with **Unsloth** on **AMD ROCm** so the model predicts the winner’s next **move** or **switch** from **raw [Pokémon Showdown](https://pokemonshowdown.com/) replay logs**. Intended for **AMD ROCm AI Developer Hub** review and publication.

## Contents

| Path | Purpose |
|------|---------|
| `pokemon_llm_agent_unsloth_rocm_tutorial.ipynb` | Main tutorial (streaming data, short SFT, inference, legality checks, mini-eval) |
| `pokemon_agent_demo_notebook_v2.ipynb` | Companion / pointers to the main notebook |
| `scripts/eval/showdown_agent_eval.py` | Shared metrics, log parsing, `build_test_samples`, batch eval |
| `scripts/eval/eval_showdown_agent.py` | CLI evaluation for merged checkpoints |
| `docs/AMD_HUB_SUBMISSION.md` | Hub submission notes + Confluence guideline link |

## Prerequisites

- AMD ROCm-capable GPU and a Python environment where **PyTorch (ROCm)** is already installed.
- **Unsloth** built or installed for your ROCm stack (follow [Unsloth](https://github.com/unslothai/unsloth) instructions).
- **Hugging Face** libraries: see `requirements.txt`. A public HF account is enough for the dataset and base model used in the notebook; use `huggingface-cli login` if you hit rate limits.

## Data and base model

- Replay corpus (streaming in the notebook): [`milkkarten/pokemon-showdown-replays-merged`](https://huggingface.co/datasets/milkkarten/pokemon-showdown-replays-merged)
- Base model: [`Qwen/Qwen3-4B`](https://huggingface.co/Qwen/Qwen3-4B)

## How to run

1. Clone this repository and open Jupyter with the **working directory set to the repo root** (the notebook discovers `scripts/eval/showdown_agent_eval.py` from there).
2. Run `pokemon_llm_agent_unsloth_rocm_tutorial.ipynb` top to bottom after your ROCm + Unsloth install cells succeed.
3. Optional: run `python3 scripts/eval/eval_showdown_agent.py --model_path … --samples 50` on a merged checkpoint for the same headline metrics as the notebook mini-eval.

## Evaluation protocol (summary)

The tutorial reports **valid format**, **type match**, and **exact match** using one implementation: `eval_showdown_agent_batch` in `showdown_agent_eval.py`, documented in the notebook §7 and in the eval script docstrings.

## License

Tutorial source code in this repository is released under **Apache-2.0** (see `LICENSE`). Third-party models, datasets, and Unsloth/PyTorch remain under their respective licenses.

## Acknowledgements

- [Unsloth](https://github.com/unslothai/unsloth), [TRL](https://github.com/huggingface/trl), [Transformers](https://github.com/huggingface/transformers), [Qwen](https://huggingface.co/Qwen), Pokémon Showdown.
