# convert_and_save_encoder.py
# -*- coding: utf-8 -*-
import argparse, os, copy, torch
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
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # 1) Load original Gemma-3 decoder
    print(f"[INFO] Loading source model: {args.src}")
    dec = AutoModelForCausalLM.from_pretrained(args.src, torch_dtype="auto")
    tok = AutoTokenizer.from_pretrained(args.src, use_fast=True)

    # 2) Create encoder-only model with the *text* config
    text_cfg = _extract_text_config(dec)
    enc = Gemma3EncoderForMaskedLM(text_cfg)  # this class sets bidirectional + no cache internally

    # 3) Align dtype to avoid load_state_dict dtype mismatch
    ref_dtype = next((p for p in dec.model.parameters()), None).dtype if hasattr(dec, "model") \
                else next(dec.parameters()).dtype
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

    # 7) Save encoder + tokenizer
    print(f"[INFO] Saving encoder checkpoint to: {args.out}")
    enc.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print("[INFO] Done.")


if __name__ == "__main__":
    main()

"""python convert_and_save_encoder.py \
  --src google/gemma-3-270m \
  --out ./gemma3-270m-encoder-bidir-model \
  --add-mask-token
"""