# AMD ROCm AI Developer Hub — tutorial submission notes

This folder is a **minimal publishable slice** of the Pokémon Showdown LLM agent tutorial (Unsloth + ROCm).

**Authors:** Yue Yuan ([@yueyuan](https://github.com/yueyuan), yueyuan@amd.com), Bill He ([@billishyahao](https://github.com/billishyahao), bill.he@amd.com).

If you publish this slice to a personal or team GitHub repo, you can add the following trailers to a commit message so GitHub lists both authors:

```text
Co-authored-by: billishyahao <bill.he@amd.com>
Co-authored-by: yueyuan <yueyuan@amd.com>
```

## Official workflow (not a generic “fork”)

Per Hub guidelines, tutorials land in the internal repo **`ROCm/gpuaidev-internal`**: clone it (**ROCm org members** can work directly from a feature branch), use `tutorial/<short-descriptor>`, add the `.ipynb` where maintainers specify, update the **root `README.md`** and **`docs/sphinx/_toc.yml.in`**, push, open a **PR** to `main`, and add **Mahdi-CV** as reviewer.

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

## Push to GitHub (maintainer)

From this repository root, after creating an **empty** public (or internal) repo on GitHub:

```bash
git remote add origin https://github.com/<org>/<repo>.git
git push -u origin main
```

Replace the URL with SSH if your team requires it (`git@github.com:<org>/<repo>.git`).
