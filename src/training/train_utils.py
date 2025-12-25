import torch
import torch.nn.functional as F

def compute_loss(logits, labels):
    
    vocab_size = logits.size(-1)

    loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        labels.view(-1),
        ignore_index=-100
    )
    return loss


def compute_accuracy(logits, labels):
    preds = logits.argmax(dim=-1)
    mask = labels != -100

    correct = (preds == labels) & mask
    return correct.sum().item() / mask.sum().item()



def clip_gradients(model, max_norm):
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)



def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}
