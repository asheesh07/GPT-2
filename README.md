# GPT-2 From Scratch — Architecture, Training, and Failure Modes

A from-scratch PyTorch implementation of GPT-2, based on
Radford et al., “Language Models are Unsupervised Multitask Learners”.

This project was built to understand transformer internals and training failure modes under real compute constraints, not to chase fluent text generation or benchmark results.

Why This Project Exists

Most modern LLM workflows hide the transformer core behind high-level libraries.
While practical, this obscures critical behaviors such as:

why loss curves can be misleading

how causal masking enables autoregression

why attention heads contribute unevenly

how repetition and memorization emerge during training

I built GPT-2 from scratch to remove these abstractions and observe these behaviors directly.

Model Architecture

This implementation follows the GPT-2 design closely:

Decoder-only transformer

Multi-head causal self-attention

Learned positional embeddings

Pre-LayerNorm residual blocks

Feed-forward MLP with GELU

Autoregressive next-token prediction objective

Architecture Flow
Input Tokens
  → Token Embedding + Positional Embedding
  → [Transformer Block × N]
      - Causal Multi-Head Self-Attention
      - Residual + LayerNorm
      - Feed-Forward MLP (GELU)
      - Residual + LayerNorm
  → Final LayerNorm
  → Linear projection to vocabulary logits

Supported Configurations

The codebase is architecturally configurable to support multiple GPT-2 scales.

Variant	Layers	Heads	Embedding Dim	Params
Tiny	4	4	256	~10M
Small	12	12	768	~117M
Medium*	24	16	1024	~345M
Large*	36	20	1280	~762M

* Medium and Large variants are architecturally supported but not trained, due to compute constraints.

Attention Experiments

The model supports pluggable attention modules for controlled experimentation:

Standard multi-head causal attention

Linear attention variants

Gated attention mechanisms

These were used to study:

training stability

memory usage

generation behavior under constrained data

Training Setup

Dataset: ~230 MB children’s stories corpus

Tokenizer: Byte Pair Encoding (BPE), vocab size 50,257

Model trained: GPT-2 Tiny (256-d, 4 layers, 4 heads)

Training steps: ~2,000

Hardware: Single consumer GPU

Objective: Autoregressive language modeling

The goal was behavioral understanding, not convergence or state-of-the-art results.

Observed Training Behavior

Training surfaced expected and instructive failure modes.

Loss–Generation Mismatch

Training loss continued to decrease even as generation quality degraded.

Rapid Overfitting

Large model capacity paired with limited data led to memorization within a few thousand steps.

Entropy Collapse & Repetition

Autoregressive decoding reinforced high-probability token paths, producing loops such as:

wave wave wave wave wave wave ...


This behavior is a known failure mode, not a bug.

Attention Head Inequality

A small subset of attention heads contributed disproportionately to useful behavior.

Sample Generation

Prompt

They wave to each other


Output (mid-training)

They wave to each other each to wave each each to other each wave each other ...


This repetition illustrates entropy collapse under limited data and compute.

What This Project Demonstrates

This project demonstrates the ability to:

implement GPT-style transformers without high-level abstractions

reason about training dynamics beyond surface metrics

identify and explain common LLM failure modes

design modular architectures for experimentation

make principled decisions under compute constraints

Limitations

Partial training (~2K steps)

Small dataset by modern standards

No distributed or mixed-precision training

Not intended for production inference

These limitations are intentional and instructive, aligned with the project’s goals.

Why This Matters

Understanding these behaviors directly informed later work on:

Retrieval-Augmented Generation (RAG)

constrained decoding strategies

agentic systems

evaluation beyond loss curves

Modern LLM systems succeed not because transformers are “magic,” but because they are augmented with retrieval, constraints, tooling, and scale.

Running the Code
pip install -r requirements.txt
python scripts/train.py --config configs/gpt2_tiny.yaml


Generate text:

python scripts/generate.py --prompt "Once upon a time"

References

Radford et al., Language Models are Unsupervised Multitask Learners

Vaswani et al., Attention Is All You Need

Final Note

This repository is not a benchmark model.

It is a mechanically faithful GPT-2 implementation built to expose how transformers actually behave when they train, overfit, and fail.

That understanding is the foundation for building robust modern LLM systems.
