# Developing Android Applications for Running Fine-Tuned LLMs Locally on Mobile Devices

This repository contains my work on building and comparing Android applications that run fine-tuned LLMs fully on-device.

The goal is to enable private, low-latency, offline AI on mobile phones by evaluating different local inference stacks.

## Project Overview

I developed and tested three Android app pipelines for local LLM inference:

- ExecuTorch app: Runs optimized `.pte` models using PyTorch ExecuTorch runtime.
- MediaPipe Tasks app: Runs `.task` models with Google's MediaPipe GenAI stack.
- llama.cpp app: Runs quantized `.gguf` models with CPU-focused inference.

This project compares speed, efficiency, and output quality across these approaches under mobile constraints.

## System Architecture

Each app follows the same high-level flow:

1. User enters a prompt in Android UI.
2. Prompt is tokenized and passed to local model runtime.
3. Model generates output tokens on-device.
4. Tokens are detokenized and shown in the chat UI.

No cloud server is required during inference.

## Models Used

I used fine-tuned versions of the following models for on-device inference:

| Model | Parameters | Speed | Output Quality |
|---|---:|---|---|
| Gemma 3 | 270M | Fast | High |
| Qwen 3 | 0.6B | Fast | High |
| SmolLM2 | 135M | Very Fast | Lower |

## Key Results

- Gemma3 270M and Qwen3 0.6B both performed well.
- Gemma3 270M delivered near Qwen-level accuracy with fewer parameters.
- llama.cpp was the most optimized stack for CPU-based local inference.
- ExecuTorch was efficient, but less optimized than llama.cpp for this setup.
- MediaPipe pipeline prioritized ease of deployment over raw performance.
- Q4-quantized variants were roughly 2x faster than BF16 variants.
- Gemma3 270M Q4 achieved the highest observed speed (~4.65 tokens/s).

## Repository Structure

- `Finetuned-inference-for-android/`: Native Android (Kotlin + Compose + ExecuTorch) app.
- `flutter_application_1/`: Flutter + MediaPipe local inference app.
- `new_gguf_llm_app/`: Flutter + llama.cpp (GGUF) local inference app.

## Why This Work Matters

- Privacy: all inference is performed on-device.
- Offline support: app works without internet once models are available locally.
- Lower latency: no network round-trip for generation.
- Practical trade-off study: speed vs quality vs memory for mobile deployment.

## References

- https://unsloth.ai/docs/get-started/fine-tuning-llms-guide
- https://unsloth.ai/docs/basics/inference-and-deployment/deploy-llms-phone
- https://ai.google.dev/gemma/docs/conversions/hf-to-mediapipe-task
- https://github.com/ggml-org/llama.cpp