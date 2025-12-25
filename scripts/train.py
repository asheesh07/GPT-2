import torch
import argparse
from src.model.decoder import GPT2
from src.config.config import GPT2_TINY
from src.training.dataloader import Train_dataloader

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_iters", type=int, default=5000)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--device", type=str, default="cpu")  # use mps later
    parser.add_argument("--checkpoint", type=str, default=None)
    return parser.parse_args()

def main():
    args = get_args()
    device = args.device

    train_loader = Train_dataloader(
        path="data/processed/train_tokens.pt",
        block_size=args.block_size,
        batch_size=args.batch_size,
        num_workers=2,
        shuffle=True
    )

    val_loader = Train_dataloader(
        path="data/processed/val_tokens.pt",
        block_size=args.block_size,
        batch_size=args.batch_size,
        num_workers=2,
        shuffle=True
    )

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

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    use_amp = False  # no cuda
    scaler = None

    for step in range(args.max_iters):

            batch = next(iter(train_loader))
            x = batch["input_ids"].to(device)
            y = batch["labels"].to(device)

            model.train()
            print("model activated")
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1)
            )
            print('model trained')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % args.eval_interval == 0:
                model.eval()
                batch = next(iter(val_loader))
                vx = batch["input_ids"].to(device)
                vy = batch["labels"].to(device)

                with torch.no_grad():
                    v_logits = model(vx)
                    v_loss = torch.nn.functional.cross_entropy(
                        v_logits.view(-1, v_logits.size(-1)),
                        vy.view(-1)
                    )

                print(f"Step {step}: Train {loss.item():.4f} | Val {v_loss.item():.4f}")
                torch.save(model.state_dict(), f"checkpoints/checkpoint_step_{step}.pt")

    print("[DONE] Training complete.")

if __name__ == "__main__":
    main()
