# convert_and_save_encoder.py
# -*- coding: utf-8 -*-
import argparse, os, copy, shutil, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig

from gemma3_biencoder import Gemma3EncoderForMaskedLM


def _extract_text_config(dec_model) -> Gemma3TextConfig:
    """
    Get a Gemma3TextConfig from the loaded decoder.
    Works for text-only and potential multi-modal wrappers.
    """
    # Prefer the inner backbone's config if present
    inner_cfg = getattr(getattr(dec_model, "model", dec_model), "config", dec_model.config)

    if isinstance(inner_cfg, Gemma3TextConfig):
        return copy.deepcopy(inner_cfg)

    # Some configs expose a .text_config (multi-modal style)
    top_cfg = getattr(dec_model, "config", inner_cfg)
    if hasattr(top_cfg, "text_config") and isinstance(top_cfg.text_config, Gemma3TextConfig):
        return copy.deepcopy(top_cfg.text_config)

    # Best-effort fallback: construct from dict
    return Gemma3TextConfig.from_dict(inner_cfg.to_dict())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="google/gemma-3-270m",
                    help="Source Gemma-3 decoder (e.g., HF hub id or local path)")
    ap.add_argument("--out", required=True, help="Where to save the encoder checkpoint")
    ap.add_argument("--add-mask-token", action="store_true",
                    help="Add <mask> to tokenizer vocab if missing (recommended for MLM)")
    ap.add_argument("--hub-format", action="store_true",
                    help="Save in Hub-ready format (adds auto_map, model code, __init__.py)")
    # Model config overrides
    ap.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto",
                    help="Model dtype (default: auto, inherits from source)")
    ap.add_argument("--sliding-window", type=int, default=None,
                    help="Override sliding window size (e.g., 512)")
    ap.add_argument("--max-position-embeddings", type=int, default=None,
                    help="Override max position embeddings / context length (e.g., 2048)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Determine target dtype
    if args.dtype == "float32":
        target_dtype = torch.float32
    elif args.dtype == "float16":
        target_dtype = torch.float16
    elif args.dtype == "bfloat16":
        target_dtype = torch.bfloat16
    else:
        target_dtype = None  # will inherit from source

    # 1) Load original Gemma-3 decoder
    print(f"[INFO] Loading source model: {args.src}")
    dec = AutoModelForCausalLM.from_pretrained(args.src, torch_dtype="auto")
    tok = AutoTokenizer.from_pretrained(args.src, use_fast=True)

    # 2) Create encoder-only model with the *text* config
    text_cfg = _extract_text_config(dec)
    
    # Apply config overrides before creating model
    if args.sliding_window is not None:
        text_cfg.sliding_window = args.sliding_window
        print(f"[INFO] Set sliding_window = {args.sliding_window}")
    if args.max_position_embeddings is not None:
        text_cfg.max_position_embeddings = args.max_position_embeddings
        print(f"[INFO] Set max_position_embeddings = {args.max_position_embeddings}")
    
    enc = Gemma3EncoderForMaskedLM(text_cfg)  # this class sets bidirectional + no cache internally

    # 3) Determine final dtype
    if target_dtype is not None:
        ref_dtype = target_dtype
        print(f"[INFO] Using dtype: {target_dtype}")
    else:
        ref_dtype = next((p for p in dec.model.parameters()), None).dtype if hasattr(dec, "model") \
                    else next(dec.parameters()).dtype
        print(f"[INFO] Inheriting dtype from source: {ref_dtype}")
    enc.to(ref_dtype)

    # 4) Copy backbone (embed + transformer layers) from decoder into encoder
    print("[INFO] Copying transformer weights into encoder…")
    src_state = dec.model.state_dict() if hasattr(dec, "model") else dec.state_dict()
    enc.encoder.load_state_dict(src_state, strict=True)

    # 5) (Optional) Add a <mask> token and ensure pad token is set
    if args.add_mask_token and tok.mask_token is None:
        print("[INFO] Adding <mask> token to tokenizer…")
        tok.add_special_tokens({"mask_token": "<mask>"})
        # Having a pad token is helpful even if you mostly use packed blocks
        if tok.pad_token is None and tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        # Resize embeddings (ties will be restored automatically)
        enc.resize_token_embeddings(len(tok))

    # 6) Sanity: lm_head is tied to input embeddings
    try:
        assert enc.lm_head.weight.data_ptr() == enc.get_input_embeddings().weight.data_ptr()
    except AssertionError:
        # Re-tie defensively (should already be tied by post_init/resize)
        enc.tie_weights()
        assert enc.lm_head.weight.data_ptr() == enc.get_input_embeddings().weight.data_ptr()

    # 7) Configure for Hub if requested
    if args.hub_format:
        enc.config.auto_map = {
            "AutoModel": "modeling_gemma3_biencoder.Gemma3EncoderForMaskedLM",
            "AutoModelForMaskedLM": "modeling_gemma3_biencoder.Gemma3EncoderForMaskedLM",
        }
        enc.config.architectures = ["Gemma3EncoderForMaskedLM"]

    # 8) Save encoder + tokenizer
    print(f"[INFO] Saving encoder checkpoint to: {args.out}")
    enc.save_pretrained(args.out, safe_serialization=True)
    tok.save_pretrained(args.out)

    # 9) Copy model code for Hub-ready format
    if args.hub_format:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        src_model_file = os.path.join(script_dir, "gemma3_biencoder.py")
        dst_model_file = os.path.join(args.out, "modeling_gemma3_biencoder.py")
        if os.path.exists(src_model_file):
            shutil.copy2(src_model_file, dst_model_file)
            print(f"[INFO] Copied modeling_gemma3_biencoder.py")
        
        # Create __init__.py
        init_file = os.path.join(args.out, "__init__.py")
        with open(init_file, "w") as f:
            f.write("from .modeling_gemma3_biencoder import Gemma3EncoderForMaskedLM\n")
        print(f"[INFO] Created __init__.py")

    print("[INFO] Done.")


if __name__ == "__main__":
    main()

"""
# Basic conversion (for local training):
python convert_and_save_encoder.py \
  --src google/gemma-3-270m \
  --out ./gemma3-270m-encoder-bidir-model \
  --add-mask-token

# Hub-ready format (can upload directly):
python convert_and_save_encoder.py \
  --src google/gemma-3-270m \
  --out ./gemma3-270m-encoder-bidir-model \
  --add-mask-token \
  --hub-format

# RexGemma-style (fp32, sliding window 512, 2048 context):
python convert_and_save_encoder.py \
  --src google/gemma-3-270m \
  --out ./gemma3-270m-encoder-sw512 \
  --add-mask-token \
  --dtype float32 \
  --sliding-window 512 \
  --max-position-embeddings 2048 \
  --hub-format
"""