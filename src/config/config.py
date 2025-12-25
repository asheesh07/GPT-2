class GPTConfig:
    """
    Base configuration for GPT-style models.
    You can subclass this to create multiple variants (Tiny, Small, etc.)
    """

    # Core model parameters
    vocab_size = 50257
    embed_size = 512
    num_layers = 6
    heads = 8
    forward_expansion = 4
    max_length = 512
    dropout = 0.1


    attention_type = "multi"


    batch_size = 16
    block_size = 128
    learning_rate = 3e-4
    weight_decay = 0.01
    betas = (0.9, 0.95)
    epochs = 3
    device = "cpu"  # or "cpu"


    train_path = "data/processed/train_tokens.pt"
    val_path = "data/processed/val_tokens.pt"


class GPT2_TINY(GPTConfig):
    """Great for your small stories dataset"""
    embed_size = 256
    num_layers = 4
    heads = 4
    max_length = 256


class GPT2_SMALL(GPTConfig):
    """Close to GPT-2 Small but lighter"""
    embed_size = 768
    num_layers = 12
    heads = 12
    max_length = 1024


class GPT2_MEDIUM(GPTConfig):
    embed_size = 1024
    num_layers = 24
    heads = 16
    max_length = 1024


class GPT2_LARGE(GPTConfig):
    embed_size = 1280
    num_layers = 36
    heads = 20
    max_length = 1024
