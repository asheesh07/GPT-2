from torch.optim import AdamW

# Optional: Lion optimizer support
try:
    from lion_pytorch import Lion
    LION_AVAILABLE = True
except:
    LION_AVAILABLE = False


def build_optimizer(model, cfg):
    if cfg.name.lower() == "adamw":
        return AdamW(
            model.parameters(),
            lr=cfg.lr,
            betas=(cfg.beta1, cfg.beta2),
            weight_decay=cfg.weight_decay
        )

    elif cfg.name.lower() == "lion":
        if not LION_AVAILABLE:
            raise ImportError("Lion optimizer not installed. pip install lion-pytorch")
        return Lion(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay
        )

    else:
        raise ValueError(f"Unknown optimizer: {cfg.name}")
