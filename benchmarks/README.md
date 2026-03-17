# Benchmarks

This directory contains benchmark suites for evaluating different model families and infrastructure components in vLLM-Omni. Each subfolder targets a different benchmark family with its own scripts, configs, and metrics. See the per-subfolder READMEs for detailed usage.

## Benchmark families

### [Qwen3-Omni](qwen3-omni/README.md) — Multimodal LLM (speech + text)

End-to-end benchmark for the Qwen3-Omni MoE model, comparing HF Transformers (offline) against the vLLM-Omni multi-stage pipeline.

- **Layout**: `qwen3-omni/transformers/` (HF baseline), `qwen3-omni/vllm_omni/` (pipeline)
- **Dataset**: SeedTTS top-100 prompts (see `build_dataset/`)
- **Key metrics**: `overall_tps`, per-stage `*_tps_avg`, latency distribution via `*.stats.jsonl`

### [Qwen3-TTS](qwen3-tts/README.md) — Text-to-Speech

Benchmarks for Qwen3-TTS models (0.6B and 1.7B variants), including online serving and async streaming modes.

- **Layout**: `qwen3-tts/transformers/` (HF baseline), `qwen3-tts/vllm_omni/` (serving + async streaming)
- **Dataset**: 12 built-in English test prompts hardcoded in the benchmark scripts; cycled to reach the desired request count (default 50)
- **Key metrics**: TTFP (time to first audio packet), E2E latency, RTF (real-time factor), throughput (audio seconds / wall-clock second)

### [Diffusion](diffusion/README.md) — Image and Video Generation

Online-serving benchmark for diffusion models, sending requests to the vLLM OpenAI-compatible endpoint.

- **Tasks**: text-to-image, text-to-video, image-to-image, image-to-video
- **Datasets**: `vbench`, `trace`, `random`
- **Key metrics**: throughput, latency percentiles, SLO attainment

### [Distributed](distributed/omni_connectors/README.md) — Cross-Node Transfer

RDMA testing for cross-node distributed transfers using MooncakeTransferEngineConnector.

- **Transfer modes**: copy, zerocopy, GPU (GPUDirect)
- **Supports**: single-node and multi-node testing

## Adding a new benchmark

1. Create a subfolder under `benchmarks/<name>/` with scripts and a `README.md`.
2. If comparing against HF Transformers, use a `transformers/` + `vllm_omni/` sub-layout (see `qwen3-omni/` or `qwen3-tts/` for examples).
3. Add shared data utilities to `build_dataset/` if applicable.
4. Update this README with a link to the new benchmark family.
