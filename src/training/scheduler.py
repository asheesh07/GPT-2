from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

def linear_warmup_decay(optimizer, warmup_steps, total_steps):

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return max(
            0.0,
            float(total_steps - step) / float(max(1, total_steps - warmup_steps))
        )
    return LambdaLR(optimizer, lr_lambda)


def build_scheduler(optimizer, cfg):
    

    name = cfg.name.lower()

    if name == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=cfg.t_max
        )

    elif name == "linear":
        return linear_warmup_decay(
            optimizer,
            warmup_steps=cfg.warmup_steps,
            total_steps=cfg.total_steps
        )

    elif name == "none":
        return None

    else:
        raise ValueError(f"Unknown scheduler: {cfg.name}")
