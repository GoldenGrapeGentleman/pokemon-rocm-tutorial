# AMD ROCm AI Developer Hub — tutorial submission notes

This folder is a **minimal publishable slice** of the Pokémon Showdown LLM agent tutorial (Unsloth + ROCm).

## Internal guidelineContribution and publication process (Confluence, AMD network):

https://amd.atlassian.net/wiki/spaces/AIG/pages/835501680/ROCm+AI+Developer+Hub+Tutorials+Contribution+and+Publication+Guidelines

If that page requires VPN, use the same checklist your team uses for other Hub tutorials (license, README, runnable steps, hardware notes).

## What reviewers should run

1. Open `pokemon_llm_agent_unsloth_rocm_tutorial.ipynb` from the **repository root** (so `scripts/eval/showdown_agent_eval.py` resolves in §7).
2. Complete environment setup cells (ROCm + Unsloth per your standard image).
3. Optional companion: `pokemon_agent_demo_notebook_v2.ipynb`.

## Full checkpoint evaluation

From repo root, after training or with a merged model path:

```bash
python3 scripts/eval/eval_showdown_agent.py --model_path /path/to/merged --samples 50
```

Metrics match the notebook mini-eval (`showdown_agent_eval.eval_showdown_agent_batch`).
