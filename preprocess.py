import os
import json
import torch
import tiktoken
import re

RAW_PATH = "data/raw/dataset.txt"
PROC_DIR = "data/processed"

FULLTEXT_PATH = f"{PROC_DIR}/full_text.txt"
TRAIN_TOKENS_PATH = f"{PROC_DIR}/train_tokens.pt"
VAL_TOKENS_PATH = f"{PROC_DIR}/val_tokens.pt"
METADATA_PATH = f"{PROC_DIR}/metadata.json"

TRAIN_SPLIT = 0.9
EOT = "<|endoftext|>"

def normalise_text(text):
    text=text.replace("\r\n","\n")
    
    text=text.replace("\xa0"," ")
    
    text=re.sub(r"[ \t]+$","",text,flags=re.MULTILINE)
    
    text=re.sub(r" {2,}","",text)
    
    return text.strip()

def load_stories(path: str):
    """
    Split raw file into paragraphs and group every 3 paragraphs as one story.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    raw=normalise_text(raw)
    
    block=re.split(r"\n\x*\n\s*\n+",raw)
    
    stories=[]
    for blk in block:
        blk=blk.strip()
        if not blk:
            continue
        blk=re.sub(r"\n\s*\n*","\n\n",blk)
        
        stories.append(blk)
    
    return stories


def main():
    os.makedirs(PROC_DIR, exist_ok=True)

    stories = load_stories(RAW_PATH)
    print(f"[INFO] Loaded {len(stories)} stories.")

    full_text = EOT.join(stories)

    with open(FULLTEXT_PATH, "w", encoding="utf-8") as f:
        f.write(full_text)
    print(f"[INFO] Wrote combined text -> {FULLTEXT_PATH}")

    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab
    print("Vocab size =", vocab_size)
    tokens = enc.encode(full_text,allowed_special={"<|endoftext|>"})
    total_tokens = len(tokens)
    print(f"[INFO] Total tokens: {total_tokens}")

    split_idx = int(total_tokens * TRAIN_SPLIT)
    train_tokens = tokens[:split_idx]
    val_tokens = tokens[split_idx:]

    torch.save(torch.tensor(train_tokens, dtype=torch.long), TRAIN_TOKENS_PATH)
    torch.save(torch.tensor(val_tokens, dtype=torch.long), VAL_TOKENS_PATH)
    print(f"[INFO] Saved train tokens -> {TRAIN_TOKENS_PATH}")
    print(f"[INFO] Saved val tokens   -> {VAL_TOKENS_PATH}")

    metadata = {
        "num_stories": len(stories),
        "total_tokens": total_tokens,
        "train_tokens": len(train_tokens),
        "val_tokens": len(val_tokens),
        "split_ratio": TRAIN_SPLIT,
        "tokenizer": "gpt2",
        "special_token": EOT,
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[INFO] Wrote metadata -> {METADATA_PATH}")
    print("[DONE] Preprocessing completed successfully.")

if __name__ == "__main__":
    main()
