# Pokémon LLM Agent with Unsloth (ROCm tutorial)

End-to-end **Jupyter** walkthrough for fine-tuning **`Qwen/Qwen3-4B`** with **Unsloth** on **AMD ROCm** so the model predicts the winner’s next **move** or **switch** from **raw [Pokémon Showdown](https://pokemonshowdown.com/) replay logs**. Intended for **AMD ROCm AI Developer Hub** review and publication.

## Contents

| Path | Purpose |
|------|---------|
| `pokemon_llm_agent_unsloth_rocm_tutorial.ipynb` | Main tutorial (Showdown rules demo, checkpoint comparison, step-by-step dataset processing, SFT, GRPO reward design, inference, mini-eval); on **AMD AI Developer Hub** docs this file is shipped **without** a local `scripts/` tree — Step 0 clones this repo shallow for eval helpers |
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

1. Prefer the latest **AMD `rocm/pytorch`** image on [Docker Hub](https://hub.docker.com/r/rocm/pytorch/tags) or an **AMD Developer Cloud** Jupyter kernel backed by **HIP-enabled PyTorch** — **not** a CPU-only PyPI `pip install torch` kernel on an MI300 partition.
2. Clone this repo and open Jupyter (from any sane working directory). **§0** of the notebook will detect an existing checkout or **`git clone --depth 1`** into `./pokemon-rocm-tutorial-extras` by default; override **`POKEMON_ROCM_EXTRAS_REPO`** / **`POKEMON_ROCM_EXTRAS_DIR`** if needed.
3. Run `pokemon_llm_agent_unsloth_rocm_tutorial.ipynb` top to bottom after ROCm + Unsloth install cells succeed.
4. Optional: run `python3 scripts/eval/eval_showdown_agent.py --model_path … --samples 50` on a merged checkpoint for the same headline metrics as the notebook mini-eval (from this repo root after clone).

## Evaluation protocol (summary)

The tutorial reports **valid format**, **type match**, and **exact match** using one implementation: `eval_showdown_agent_batch` in `showdown_agent_eval.py`, documented in the notebook §7 and in the eval script docstrings.

## License

Tutorial source code in this repository is released under **Apache-2.0** (see `LICENSE`). Third-party models, datasets, and Unsloth/PyTorch remain under their respective licenses.

## Authors

- **Yue Yuan** — [@yueyuan](https://github.com/yueyuan) · yueyuan@amd.com
- **Bill He** — [@billishyahao](https://github.com/billishyahao) · bill.he@amd.com

## Acknowledgements

- [Unsloth](https://github.com/unslothai/unsloth), [TRL](https://github.com/huggingface/trl), [Transformers](https://github.com/huggingface/transformers), [Qwen](https://huggingface.co/Qwen), Pokémon Showdown.
