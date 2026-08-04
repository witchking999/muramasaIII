# Submodules Guide

This document describes every submodule in the `submodules/` directory, what it does, and how an AI agent should work with it.

---

## Table of Contents

1. [NV-Tesseract](#1-nv-tesseract)
2. [Automodel (NeMo AutoModel)](#2-automodel-nemo-automodel)
3. [MoME](#3-mome)
4. [Megatron-Bridge (NeMo Megatron Bridge)](#4-megatron-bridge-nemo-megatron-bridge)
5. [TokenCast](#5-tokencast)
6. [runpod-mcp](#6-runpod-mcp)
7. [Chem3DLLM](#7-chem3dllm)
8. [Working with Submodules](#working-with-submodules)

---

## 1. NV-Tesseract

| Field | Value |
|-------|-------|
| **Path** | `submodules/NV-Tesseract` |
| **Upstream** | https://github.com/NVIDIA/NV-Tesseract |
| **Language** | Python 3.12+ |
| **License** | Apache 2.0 |

### What it is
NVIDIA Tesseract is an open-source **time-series analysis library** covering two tasks:
- **Forecasting** — multivariate time-series forecasting built on a pretrained transformer backbone, with an optional DARR (context-enhanced / kNN retrieval) mode and a model-agnostic interpretability engine that produces lag/feature attributions and PDF reports.
- **Anomaly Detection** — diffusion-based multivariate anomaly detection using NVIDIA's proprietary SCS/MACS adaptive thresholding algorithms.

### How it works
Both modules expose a single high-level SDK function:

```python
# Forecasting
from sdk.forecasting import perform_forecasting
forecasts = perform_forecasting(df=df, seq_len=512, forecast_horizon=72)

# Anomaly Detection
from sdk.anomaly_analysis import perform_anomaly_analysis_with_diffusion
results = perform_anomaly_analysis_with_diffusion(df=df, threshold_strategy="scs")
```

Pretrained model weights are auto-downloaded from Hugging Face on first run. GPU (CUDA or Apple MPS) is recommended but the library falls back to CPU.

### Agent usage notes
- Install each module independently: `cd forecasting && pip install -e .` or `cd ad_diffusion && pip install -e .`
- Pass a `pandas.DataFrame` with a `timestamp` column plus numeric feature columns.
- Set `interpretability=True` on `perform_forecasting` to get explainability artifacts (JSON, CSV, PDF) in a timestamped output directory.
- Fine-tuning scripts live under `forecasting/examples/finetune_example.py` and `ad_diffusion/examples/finetune_example.py`.

---

## 2. Automodel (NeMo AutoModel)

| Field | Value |
|-------|-------|
| **Path** | `submodules/Automodel` |
| **Upstream** | https://github.com/NVIDIA-NeMo/Automodel |
| **Language** | Python |
| **License** | Apache 2.0 |

### What it is
NeMo AutoModel is NVIDIA's next-generation framework for **training, fine-tuning, and aligning large language and multimodal models** at scale. It is the spiritual successor to the NeMo Framework and integrates with Megatron-LM for distributed training.

### How it works
AutoModel wraps complex distributed training recipes behind a simple API and CLI. It supports supervised fine-tuning (SFT), reinforcement learning from human feedback (RLHF), direct preference optimization (DPO), and more. Models are configured via YAML recipes.

### Agent usage notes
- This is the **primary training entry-point** for any LLM or multimodal model work in this repository.
- Use in conjunction with `Megatron-Bridge` (see below) when doing large-scale distributed runs.
- Refer to its internal `README.md` for available recipes and configuration keys.

---

## 3. MoME

| Field | Value |
|-------|-------|
| **Path** | `submodules/MoME` |
| **Upstream** | https://github.com/BruceZhangReve/MoME |
| **Language** | Python 3.10 / PyTorch 2.5 |
| **License** | See repo |

### What it is
MoME (Mixture of Modulated Experts) is a research framework for **multi-modal time-series forecasting**. Instead of fusing text tokens directly with time-series patch tokens, it uses **Expert Modulation** — textual signals condition both the routing weights and the expert computations inside a Mixture-of-Experts (MoE) Transformer. This produces direct, efficient cross-modal control without requiring large text-time-series pair datasets.

### How it works
- Time series are split into patches and encoded by a configurable backbone (PatchTST, iTransformer, DLinear, etc.).
- A `QueryPool` module extracts instruction tokens from a pretrained LLM (default: Qwen1.5-MoE-A2.7B).
- The instruction tokens modulate both the expert routing (`--router_modulation`) and expert weight matrices (`--modulation`).
- Training and evaluation are driven by `train.py` / `evaluate.py` with dataset-specific flags.

### Agent usage notes
- Create the conda environment from `environment.yaml` before doing anything else.
- Download pretrained LLMs (GPT2, Qwen1.5-MoE-A2.7B) from HuggingFace and place them under `./llm/`.
- Key hyperparameters: `--n_experts 4 --topk 2 --instructor_query 3 --lambda_e 0.75`.
- Use `--use_bfloat16` to fit training into a single 48 GB GPU.
- Pass `--return_expert_selection --eval_mode expert_selection` to analyze which experts activate for different text prompts.

---

## 4. Megatron-Bridge (NeMo Megatron Bridge)

| Field | Value |
|-------|-------|
| **Path** | `submodules/Megatron-Bridge` |
| **Upstream** | https://github.com/NVIDIA-NeMo/Megatron-Bridge |
| **Language** | Python |
| **License** | Apache 2.0 |

### What it is
NeMo Megatron Bridge is a **compatibility and interoperability layer** between NVIDIA's NeMo framework and Megatron-LM. It exposes Megatron's high-performance, tensor/pipeline/sequence-parallel training kernels to NeMo model recipes without requiring users to write low-level Megatron code directly.

### How it works
The library provides adapter classes and utility functions that translate between NeMo model configurations and Megatron's internal distributed training state. It handles model parallelism topology, checkpoint conversion between the two formats, and optimizer state sharding.

### Agent usage notes
- Treat this as a **low-level dependency** rather than a user-facing tool — it is consumed by `Automodel` (above).
- Use it directly only when you need to convert checkpoints between Megatron and NeMo formats, or when debugging distributed training topology issues.
- Requires a correctly configured NCCL / GPU cluster environment.

---

## 5. TokenCast

| Field | Value |
|-------|-------|
| **Path** | `submodules/TokenCast` |
| **Upstream** | https://github.com/ustc-time-series/TokenCast |
| **Language** | Python |
| **License** | See repo |

### What it is
TokenCast is a research model for **time-series forecasting** that tokenizes time-series patches and casts them into a language-model embedding space, enabling zero-shot and few-shot temporal generalization through a token-level forecasting head.

### How it works
Input sequences are divided into fixed-length patches, each patch is projected to a token embedding, and a pretrained or fine-tuned Transformer predicts future token embeddings that are then projected back to the original time-series scale.

### Agent usage notes
- Consult the repo's internal documentation for dataset preparation scripts and training commands.
- Compatible with standard NLP-style training loops — treat each time-series window as a "sentence" of patch tokens.

---

## 6. runpod-mcp

| Field | Value |
|-------|-------|
| **Path** | `submodules/runpod-mcp` |
| **Upstream** | https://github.com/runpod/runpod-mcp |
| **Language** | TypeScript / Node.js 18+ |
| **License** | Apache 2.0 |

### What it is
The official **RunPod Model Context Protocol (MCP) server**. It lets any MCP-compatible AI agent (Claude Code, Cursor, Copilot, Windsurf, etc.) manage RunPod cloud GPU infrastructure in natural language — creating/stopping Pods, managing Serverless endpoints, network volumes, templates, and more.

### How it works
The server implements the MCP specification over HTTP (hosted at `https://mcp.getrunpod.io/`) or locally via `stdio`. Authentication is either OAuth ("Sign in with RunPod") for the hosted path, or a `RUNPOD_API_KEY` environment variable for local use. Every request proxies the caller's token to the RunPod REST API — the server never persists credentials.

### Agent usage notes
- **Hosted (recommended for agents):** Add `https://mcp.getrunpod.io/` as an MCP server URL in your client config. An OAuth flow runs once; subsequent requests are automatic.
- **Local:** `RUNPOD_API_KEY=<key> npx -y @runpod/mcp-server@latest`
- Natural-language commands work out of the box: *"Create a Pod with a 4090 GPU"*, *"List all my Serverless endpoints"*, *"Stop pod abc123"*.
- Use this submodule as the **control plane** for any workflow that needs to spin up or tear down GPU compute dynamically.
- Build: `pnpm install && pnpm build`; Test: `pnpm test` (offline, no API key needed).

---

## 7. Chem3DLLM

| Field | Value |
|-------|-------|
| **Path** | `submodules/Chem3DLLM` |
| **Upstream** | https://github.com/Joe-J/Chem3DLLM |
| **Language** | Python 3.7+ |
| **License** | MIT |

### What it is
Chem3DLLM is a **protein-conditioned multimodal LLM for structure-based drug design**. Its core contribution is the **RCMT (Reversible Compact Molecular Text)** format — a lossless, text-serializable representation of 3D molecular geometry that achieves 3× size reduction compared to raw coordinate lists (RMSD < 1e-4 Å on round-trip). This lets standard LLMs generate and reason about 3D molecular structures using their native text interface.

### How it works
- Atoms are encoded as `SYMBOL@X,Y,Z` and bonds as `ATOM1-ATOM2:ORDER`, separated by `#`.
- `sdf2text.py` converts SDF molecular files to/from RCMT.
- Training data is formatted as instruction-tuning JSON (LLaMA-Factory compatible).
- The unified architecture conditions generation on protein pocket structures alongside ligand SMILES or RCMT inputs.

### Agent usage notes
- Use `sdf_to_compact_text(mol)` / `compact_text_to_mol(text)` for all molecule I/O.
- Generate training data with `convert_qm9_sdf_to_llamafactory_json(...)` — output is a JSON array of `{instruction, input, output}` records ready for fine-tuning.
- Validate round-trip fidelity with `test_compact_roundtrip(sdf_path, index)` before any training run.
- Drug design benchmark: Vina score −7.21 (state-of-the-art on CrossDocked2020).

---

## Working with Submodules

### Initial checkout (clone with all submodules)
```bash
git clone --recurse-submodules https://github.com/witchking999/muramasaIII.git
```

### If you already cloned without `--recurse-submodules`
```bash
git submodule update --init --recursive
```

### Update a single submodule to its latest upstream commit
```bash
git submodule update --remote submodules/<name>
git add submodules/<name>
git commit -m "chore: bump <name> to latest"
```

### Update all submodules at once
```bash
git submodule update --remote --merge
```

### Pin a submodule to a specific commit or tag
```bash
cd submodules/<name>
git checkout <tag-or-sha>
cd ../..
git add submodules/<name>
git commit -m "chore: pin <name> to <tag-or-sha>"
```

### Check which commit each submodule is currently at
```bash
git submodule status
```

### Agent workflow tip
When an agent needs to use functionality from a submodule:
1. Confirm the submodule directory is populated (`ls submodules/<name>` should not be empty).
2. If empty, run `git submodule update --init submodules/<name>`.
3. Install the submodule's dependencies from within its directory (Python: `pip install -e .`; Node.js: `pnpm install`).
4. Import or invoke as documented in that submodule's section above.
