#!/usr/bin/env python3
import torch
import argparse
import tiktoken
from src.model.decoder import GPT2
from src.config.config import GPT2_TINY


def sample_next_token(logits, temperature=1.0, top_k=50):
    logits = logits / temperature

    if top_k is not None:
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[..., -1]] = -float('inf')

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main():
    args = get_args()
    device = args.device

    # tokenizer
    enc = tiktoken.get_encoding("gpt2")

    # ---- LOAD CONFIG (same as training) ----
    cfg = GPT2_TINY()

    # ---- BUILD MODEL with SAME CONFIG ----
    model = GPT2(
        vocab_size=cfg.vocab_size,
        embed_size=cfg.embed_size,
        num_layers=cfg.num_layers,
        heads=cfg.heads,
        forward_expansion=cfg.forward_expansion,
        max_length=cfg.max_length,
        dropout=cfg.dropout,
        attention_type=cfg.attention_type,
    ).to(device)

    # ---- LOAD CHECKPOINT ----
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # ---- ENCODE PROMPT ----
    input_ids = enc.encode(args.prompt)
    x = torch.tensor([input_ids], dtype=torch.long).to(device)

    # ---- AUTOREGRESSIVE GENERATION ----
    for _ in range(args.max_new_tokens):
        logits = model(x)     # [1, seq_len, vocab]
        last_logits = logits[:, -1, :]  # last token

        next_id = sample_next_token(
            last_logits.squeeze(0),
            temperature=args.temperature,
            top_k=args.top_k
        )
        next_id = next_id.unsqueeze(0)
        x = torch.cat([x, next_id], dim=1)

        # optional: print as it writes
        # print(enc.decode([next_id.item()]), end="", flush=True)

    print("\n---- GENERATED TEXT ----\n")
    print(enc.decode(x[0].tolist()))


if __name__ == "__main__":
    main()
