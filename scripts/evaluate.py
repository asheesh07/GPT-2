#!/usr/bin/env python3
import torch
from src.model.decoder import GPT2
from src.training.dataloader import Train_dataloader
from src.config.config import GPT2_TINY

def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0
    count = 0

    with torch.no_grad():
        for _ in range(50):  # evaluate on 50 random batches
            batch = next(iter(val_loader))
            x = batch["input_ids"].to(device)
            y = batch["labels"].to(device)
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1)
            )
            total_loss += loss.item()
            count += 1

    return total_loss / count

def main():
    device = "cpu"

    cfg = GPT2_TINY()

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
    model.load_state_dict(torch.load("checkpoint_step_1000.pt"))

    val_loader = Train_dataloader("data/processed/val_tokens.pt",
                                block_size=128,
                                batch_size=16,
                                num_workers=2,
                                shuffle=True)

    val_loss = evaluate(model, val_loader, device)
    print(f"Validation Loss: {val_loss:.4f}")

if __name__ == "__main__":
    main()
