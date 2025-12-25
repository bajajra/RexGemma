#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Export a Gemma3EncoderForMaskedLM checkpoint to a proper HuggingFace Hub-ready format.

This script:
  1. Loads the model & tokenizer from a training checkpoint
  2. Configures auto_map for the custom model class (enables AutoModel loading)
  3. Copies the custom model code into the output directory
  4. Generates a model card (README.md)
  5. Optionally pushes directly to the HuggingFace Hub
"""

from __future__ import annotations
import argparse
import os
import shutil
import json
import torch
from transformers import AutoTokenizer

# Your custom encoder class
from gemma3_biencoder import Gemma3EncoderForMaskedLM


def generate_model_card(
    model_name: str,
    base_model: str = "thebajajra/Gemma3-270M-encoder",
    sliding_window: int = 512,
    max_seq_len: int = 2048,
    vocab_size: int = 262145,
) -> str:
    """Generate a README.md model card for the Hub."""
    return f"""---
library_name: transformers
tags:
  - gemma3
  - gemma3_text
  - encoder
  - bidirectional
  - masked-language-modeling
  - text-embeddings
  - feature-extraction
  - custom_code
license: mit
base_model: {base_model}
pipeline_tag: fill-mask
---

# {model_name}

A **bidirectional encoder** fine-tuned from [{base_model}](https://huggingface.co/{base_model}) with Masked Language Modeling (MLM).

## Model Description

This model is a BERT-style bidirectional encoder based on Gemma 3 architecture:
- Bidirectional attention (not causal)
- Masked language modeling head (tied to input embeddings)
- Trained with 15% token masking (BERT-style MLM)

### Architecture Details

| Parameter | Value |
|-----------|-------|
| Base Model | [`{base_model}`](https://huggingface.co/{base_model}) |
| Vocab Size | {vocab_size:,} |
| Sliding Window | {sliding_window} |
| Max Sequence Length | {max_seq_len} |
| Attention | Bidirectional |

## Usage

### Loading the Model

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("{model_name}")
model = AutoModelForMaskedLM.from_pretrained("{model_name}", trust_remote_code=True)
```

### Masked Language Modeling

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline

model = AutoModelForMaskedLM.from_pretrained("{model_name}", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("{model_name}")

fill = pipeline("fill-mask", model=model, tokenizer=tokenizer)
fill("Best [MASK] headphones under $100.")
```

### Embeddings / Feature Extraction

```python
import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("{model_name}")
model = AutoModel.from_pretrained("{model_name}", trust_remote_code=True)

texts = ["wireless mouse", "ergonomic mouse pad"]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    # Mean-pool last hidden state
    attn = inputs["attention_mask"].unsqueeze(-1)
    embeddings = (outputs.last_hidden_state * attn).sum(1) / attn.sum(1)
    # Normalize for cosine similarity
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
```

### Sentence-Transformers

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

model_mlm = AutoModelForMaskedLM.from_pretrained("{model_name}", trust_remote_code=True)
encoder = model_mlm.encoder
tokenizer = AutoTokenizer.from_pretrained("{model_name}")

ENCODER_DIR = "encoder-only"
encoder.save_pretrained(ENCODER_DIR)
tokenizer.save_pretrained(ENCODER_DIR)

model = SentenceTransformer(ENCODER_DIR)
```

## Training

This model was trained using MLM on packed sequences with:
- Dynamic BERT-style token masking (15%)
- AdamW optimizer with fused kernels
- Mixed precision training

## License

MIT License

## Citation

If you use this model, please cite this repository.
"""


def main():
    ap = argparse.ArgumentParser(
        description="Export a Gemma3EncoderForMaskedLM checkpoint to HuggingFace Hub format"
    )
    ap.add_argument(
        "--checkpoint-dir",
        required=True,
        help="Path to the training checkpoint directory (e.g., ./output/checkpoint-5000)",
    )
    ap.add_argument(
        "--output-dir",
        required=True,
        help="Where to save the Hub-ready model",
    )
    ap.add_argument(
        "--model-name",
        default=None,
        help="Model name for the model card (defaults to output dir basename)",
    )
    ap.add_argument(
        "--base-model",
        default="thebajajra/Gemma3-270M-encoder",
        help="Base model name for documentation",
    )
    ap.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push to HuggingFace Hub after saving",
    )
    ap.add_argument(
        "--hub-repo-id",
        default=None,
        help="Hub repository ID (e.g., 'username/model-name'). Required if --push-to-hub is set",
    )
    ap.add_argument(
        "--private",
        action="store_true",
        help="Make the Hub repository private",
    )
    ap.add_argument(
        "--dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
        help="Data type for saving the model",
    )
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Determine model name
    model_name = args.model_name or os.path.basename(os.path.normpath(args.output_dir))
    if args.hub_repo_id:
        model_name = args.hub_repo_id

    # ---- Load tokenizer ----
    print(f"[INFO] Loading tokenizer from: {args.checkpoint_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir, use_fast=True)

    # ---- Determine dtype ----
    if args.dtype == "auto":
        torch_dtype = "auto"
    elif args.dtype == "float32":
        torch_dtype = torch.float32
    elif args.dtype == "float16":
        torch_dtype = torch.float16
    elif args.dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = "auto"

    # ---- Load model ----
    print(f"[INFO] Loading model from: {args.checkpoint_dir}")
    model = Gemma3EncoderForMaskedLM.from_pretrained(
        args.checkpoint_dir,
        torch_dtype=torch_dtype,
    )

    # Verify weight tying
    if getattr(model.config, "tie_word_embeddings", True):
        try:
            assert model.lm_head.weight.data_ptr() == model.get_input_embeddings().weight.data_ptr()
            print("[INFO] ✓ Weight tying verified")
        except AssertionError:
            print("[WARN] Re-tying weights...")
            model.tie_weights()

    # ---- Configure auto_map for custom model loading ----
    # This allows users to load with AutoModel + trust_remote_code=True
    # Following HF convention: modeling_<model_name>.py
    model.config.auto_map = {
        "AutoModel": "modeling_gemma3_biencoder.Gemma3EncoderForMaskedLM",
        "AutoModelForMaskedLM": "modeling_gemma3_biencoder.Gemma3EncoderForMaskedLM",
    }

    # Ensure architectures is set correctly
    model.config.architectures = ["Gemma3EncoderForMaskedLM"]

    # ---- Save model and tokenizer ----
    print(f"[INFO] Saving model to: {args.output_dir}")
    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)

    # ---- Copy the custom model code (renamed to HF convention) ----
    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_model_file = os.path.join(script_dir, "gemma3_biencoder.py")
    dst_model_file = os.path.join(args.output_dir, "modeling_gemma3_biencoder.py")

    if os.path.exists(src_model_file):
        shutil.copy2(src_model_file, dst_model_file)
        print(f"[INFO] ✓ Copied to modeling_gemma3_biencoder.py (HF convention)")
    else:
        print(f"[WARN] Could not find {src_model_file} - you may need to copy it manually")

    # ---- Create __init__.py for proper module import ----
    init_file = os.path.join(args.output_dir, "__init__.py")
    with open(init_file, "w") as f:
        f.write("from .modeling_gemma3_biencoder import Gemma3EncoderForMaskedLM\n")
    print(f"[INFO] ✓ Created __init__.py")

    # ---- Generate model card ----
    sliding_window = getattr(model.config, "sliding_window", 512)
    max_seq_len = getattr(model.config, "max_position_embeddings", 2048)
    vocab_size = getattr(model.config, "vocab_size", 262145)

    readme_content = generate_model_card(
        model_name=model_name,
        base_model=args.base_model,
        sliding_window=sliding_window,
        max_seq_len=max_seq_len,
        vocab_size=vocab_size,
    )

    readme_path = os.path.join(args.output_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(readme_content)
    print(f"[INFO] ✓ Generated README.md model card")

    # ---- Print saved files ----
    print("\n[INFO] Saved files:")
    for fname in sorted(os.listdir(args.output_dir)):
        fpath = os.path.join(args.output_dir, fname)
        if os.path.isfile(fpath):
            size_mb = os.path.getsize(fpath) / (1024 * 1024)
            print(f"  - {fname} ({size_mb:.2f} MB)")

    # ---- Push to Hub if requested ----
    if args.push_to_hub:
        if not args.hub_repo_id:
            print("[ERROR] --hub-repo-id is required when using --push-to-hub")
            return

        print(f"\n[INFO] Pushing to HuggingFace Hub: {args.hub_repo_id}")
        model.push_to_hub(
            args.hub_repo_id,
            private=args.private,
            safe_serialization=True,
        )
        tokenizer.push_to_hub(
            args.hub_repo_id,
            private=args.private,
        )

        # Push additional files (custom model code, __init__, README)
        from huggingface_hub import HfApi
        api = HfApi()
        
        if os.path.exists(dst_model_file):
            api.upload_file(
                path_or_fileobj=dst_model_file,
                path_in_repo="modeling_gemma3_biencoder.py",
                repo_id=args.hub_repo_id,
                repo_type="model",
            )

        # Push __init__.py
        if os.path.exists(init_file):
            api.upload_file(
                path_or_fileobj=init_file,
                path_in_repo="__init__.py",
                repo_id=args.hub_repo_id,
                repo_type="model",
            )

        # Push README
        api.upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=args.hub_repo_id,
            repo_type="model",
        )

        print(f"[INFO] ✓ Successfully pushed to: https://huggingface.co/{args.hub_repo_id}")

    print("\n[INFO] Export complete!")
    print(f"\nTo load this model:")
    print(f'  from transformers import AutoTokenizer, AutoModel')
    print(f'  tokenizer = AutoTokenizer.from_pretrained("{args.output_dir}")')
    print(f'  model = AutoModel.from_pretrained("{args.output_dir}", trust_remote_code=True)')


if __name__ == "__main__":
    main()


# Example usage:
"""
# Export locally:
python export_to_hf_hub.py \
  --checkpoint-dir ./gemma3-270m_sw256_512sql_4096tk/checkpoint-5000 \
  --output-dir ./gemma3-encoder-270m-mlm

# Export and push to Hub:
python export_to_hf_hub.py \
  --checkpoint-dir ./gemma3-270m_sw256_512sql_4096tk/checkpoint-5000 \
  --output-dir ./gemma3-encoder-270m-mlm \
  --push-to-hub \
  --hub-repo-id thebajajra/RexGemma-new

# With custom base model reference:
python export_to_hf_hub.py \
  --checkpoint-dir ./gemma3-270m_sw256_512sql_4096tk/checkpoint-5000 \
  --output-dir ./gemma3-encoder-270m-mlm \
  --base-model thebajajra/Gemma3-270M-encoder \
  --push-to-hub \
  --hub-repo-id thebajajra/RexGemma-new
"""

