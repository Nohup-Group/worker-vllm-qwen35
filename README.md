# worker-vllm-qwen35

RunPod Serverless worker for **Qwen3.5** models using vLLM.

Fork of [runpod-workers/worker-vllm](https://github.com/runpod-workers/worker-vllm) with one key change: **vLLM nightly** is installed by default because `Qwen3_5ForConditionalGeneration` and `Qwen3_5MoeForConditionalGeneration` are not yet in vLLM stable (v0.16.0).

## Supported Models

| Model | Architecture | Params (active) | VRAM needed | Min GPU |
|---|---|---|---|---|
| [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | `Qwen3_5ForConditionalGeneration` (dense) | 9B (9B) | ~18GB FP16 | 24GB |
| [Qwen3.5-35B-A3B-GPTQ-Int4](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-GPTQ-Int4) | `Qwen3_5MoeForConditionalGeneration` (MoE) | 35B (3B) | ~22GB GPTQ-Int4 | 48GB |

## What changed from upstream

| Area | Upstream (worker-vllm) | This fork |
|---|---|---|
| vLLM version | v0.16.0 stable | Nightly (has Qwen3.5 registry) |
| transformers | >= 4.57.0 (pip) | Git HEAD (Qwen3.5 config support) |
| MODEL_NAME normalization | No | Splits `repo:hash` into MODEL_NAME + MODEL_REVISION |
| Startup validation | No | Logs model config + checks architecture support |

## Quick Start

Use the **same Docker image** for both models — only the environment variables differ.

### Build the image

```bash
docker build -t your-registry/worker-vllm-qwen35:latest .
docker push your-registry/worker-vllm-qwen35:latest
```

Or deploy via GitHub repo on RunPod (Serverless > New Endpoint > GitHub Repo).

### Endpoint A: Qwen3.5-35B-A3B (MoE, GPTQ-Int4)

| Setting | Value |
|---|---|
| **GPU** | 48GB (A6000 or L40S) |
| `MODEL_NAME` | `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4` |
| `QUANTIZATION` | `gptq` |
| `MAX_MODEL_LEN` | `32768` |
| `GPU_MEMORY_UTILIZATION` | `0.92` |
| `TRUST_REMOTE_CODE` | `true` |
| `DISABLE_LOG_STATS` | `true` |

### Endpoint B: Qwen3.5-9B (Dense, FP16)

| Setting | Value |
|---|---|
| **GPU** | 24GB (L4 or RTX 4090) |
| `MODEL_NAME` | `Qwen/Qwen3.5-9B` |
| `MAX_MODEL_LEN` | `32768` |
| `GPU_MEMORY_UTILIZATION` | `0.92` |
| `TRUST_REMOTE_CODE` | `true` |
| `DISABLE_LOG_STATS` | `true` |

> No `QUANTIZATION` needed for the 9B — it fits in 24GB at FP16 (~18GB weights).

## GPU Selection

| GPU | VRAM | Price/hr | Use for |
|---|---|---|---|
| 24GB (L4, RTX 4090) | 24GB | ~$0.44-0.69 | Qwen3.5-9B |
| 48GB (A6000, L40S) | 48GB | $0.76-1.22 | Qwen3.5-35B-A3B-GPTQ-Int4 |
| 80GB (A100, H100) | 80GB | $2.49-4.49 | Either, if you need huge context |

## Disabling Thinking Mode

Qwen3.5 defaults to thinking mode enabled. For tasks like chunking where you don't need chain-of-thought, disable it:

```python
# Option 1: In the system/user message
messages = [{"role": "user", "content": "/no_think\nYour actual prompt here"}]

# Option 2: Via extra_body (OpenAI-compatible route)
response = client.chat.completions.create(
    model="Qwen/Qwen3.5-9B",
    messages=messages,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
```

## API Compatibility

Both endpoints expose the same OpenAI-compatible API as the upstream worker-vllm:

```python
import openai

# Endpoint A (35B MoE)
client_35b = openai.OpenAI(
    base_url="https://api.runpod.ai/v2/ENDPOINT_A_ID/openai/v1",
    api_key="YOUR_RUNPOD_API_KEY",
)

# Endpoint B (9B dense)
client_9b = openai.OpenAI(
    base_url="https://api.runpod.ai/v2/ENDPOINT_B_ID/openai/v1",
    api_key="YOUR_RUNPOD_API_KEY",
)

response = client_9b.chat.completions.create(
    model="Qwen/Qwen3.5-9B",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=256,
)
```

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

## When to re-sync with upstream

Once vLLM stable includes the Qwen3.5 architectures in its model registry (likely v0.17.0+), you can switch back to the upstream `runpod/worker-v1-vllm` image and retire this fork.
