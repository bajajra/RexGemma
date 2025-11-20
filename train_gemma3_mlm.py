#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MLM training for Gemma-3 **bidirectional encoder** (your Gemma3EncoderForMaskedLM).

Highlights:
  - Packed, fixed-length blocks (no runtime padding -> denser compute)
  - FlashAttention-2 / SDPA preference, TF32 on Ampere+
  - bf16/fp16, fused optimizer, gradient checkpointing (use_reentrant=False)
  - Throttled MLM accuracy logging (cheap, but not on every step)
  - Tokens-per-device-step knob to auto-derive batch size
  - Optionally: DeepSpeed ZeRO-3 for multi-GPU or memory-limited scenarios

Assumptions:
  * Converted encoder is at --model-dir (e.g., ./gemma3-270m-encoder-bidir)
  * Pretokenized dataset saved with datasets.save_to_disk, columns:
    - input_ids: List[int], attention_mask: List[int]
"""

from __future__ import annotations
import argparse, os, math, json
from typing import Dict, Iterator, List, Tuple, Optional
import time
import numpy as np
import torch
from datasets import load_from_disk, Dataset, DatasetDict, concatenate_datasets
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)

# Your custom encoder (bidirectional, MLM head)
from gemma3_biencoder import Gemma3EncoderForMaskedLM


# ----------------- Custom Trainer for MLM Accuracy (throttled) -----------------

class MLMAccuracyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        # Delegate forward pass (handles device move, autocast, etc.)
        try:
            loss, outputs = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch, **kwargs)
        except TypeError:
            loss, outputs = super().compute_loss(model, inputs, return_outputs=True, **kwargs)

        # # Throttle: log only every `logging_steps` on main process
        # if (
        #     self.state.is_local_process_zero
        #     and "labels" in inputs
        #     and self.state.global_step % max(1, self.args.logging_steps) == 0
        # ):
        #     logits = outputs.get("logits", None)
        #     labels = inputs.get("labels", None)
        #     if logits is not None and labels is not None:
        #         with torch.no_grad():
        #             preds = torch.argmax(logits, dim=-1)
        #             mask = labels != -100
        #             if mask.any():
        #                 acc = (preds[mask] == labels[mask]).float().mean().item()
        #                 self.log({"mlm_accuracy": acc})

        # return (loss, outputs) if return_outputs else loss
        # Throttled logging: once every logging interval on rank 0
        if (
            self.state.is_local_process_zero
            and self.state.global_step % max(1, self.args.logging_steps) == 0
        ):
            logs = {}

            # 1) MLM accuracy on masked tokens (you already had this)
            labels = inputs.get("labels")
            logits = outputs.get("logits")
            if logits is not None and labels is not None:
                with torch.no_grad():
                    preds = torch.argmax(logits, dim=-1)
                    mask = labels != -100
                    if mask.any():
                        acc = (preds[mask] == labels[mask]).float().mean().item()
                        logs["mlm_accuracy"] = acc
                        logs["mask_frac"] = mask.float().mean().item()

            # 2) Perplexity (masked-token loss)
            with torch.no_grad():
                logs["train_ppl"] = float(math.exp(loss.detach().float().item()))

            # 3) Tokens/sec (simple running window since last log)
            bs_tokens = 0
            if "input_ids" in inputs and isinstance(inputs["input_ids"], torch.Tensor):
                bs_tokens = int(inputs["input_ids"].numel())  # B*S

            if not hasattr(self, "_tps_t0"):
                self._tps_t0 = time.time()
                self._tps_tok = 0
            self._tps_tok += bs_tokens

            now = time.time()
            dt = now - self._tps_t0
            if dt > 0:
                logs["tokens_per_sec"] = self._tps_tok / dt
            # reset the window
            self._tps_t0 = now
            self._tps_tok = 0

            # 4) GPU memory (GB)
            if torch.cuda.is_available():
                logs["max_mem_reserved_gb"] = round(torch.cuda.max_memory_reserved() / (1024**3), 3)

            self.log(logs)

        return (loss, outputs) if return_outputs else loss


# ------------------------- Packing utilities -------------------------

def _as_dataset(dataset_or_dict) -> Dataset:
    """Make sure we return a single Dataset (concatenate splits if needed)."""
    if isinstance(dataset_or_dict, Dataset):
        return dataset_or_dict
    if isinstance(dataset_or_dict, DatasetDict):
        if "train" in dataset_or_dict:
            return dataset_or_dict["train"]
        # Concatenate all splits if no explicit train split
        return concatenate_datasets(list(dataset_or_dict.values()))
    raise ValueError("Unsupported dataset object loaded from disk.")

def make_packed_dataset(
    dataset_path: str,
    tokenizer,
    pack_len: int,
    out_cap: Optional[int] = None,
    num_proc: int = 8,
) -> Dataset:
    """
    Build a packed Dataset (fixed-length blocks) from an on-disk pretokenized dataset.
    We do batching+map with multiprocessing for throughput.
    """
    base = _as_dataset(load_from_disk(dataset_path))
    eos_id = tokenizer.eos_token_id or tokenizer.sep_token_id

    def pack_batch(batch: Dict[str, List]) -> Dict[str, List]:
        # 1) Concatenate trimmed examples with EOS separators
        all_tokens = []
        for ids, mask in zip(batch["input_ids"], batch["attention_mask"]):
            L = int(np.sum(mask))
            if L > 0:
                all_tokens.extend(ids[:L])
                if eos_id is not None:
                    all_tokens.append(eos_id)

        # 2) Chunk into fixed-length blocks (drop tail for tight packing)
        packed_ids = []
        for i in range(0, len(all_tokens), pack_len):
            chunk = all_tokens[i : i + pack_len]
            if len(chunk) == pack_len:
                packed_ids.append(chunk)

        if not packed_ids:
            return {"input_ids": [], "attention_mask": []}

        # 3) All ones (no intra-batch padding)
        return {
            "input_ids": packed_ids,
            "attention_mask": [[1] * pack_len for _ in packed_ids],
        }

    packed = base.map(
        pack_batch,
        batched=True,
        batch_size=1000,                  # tune to RAM/CPU
        num_proc=max(1, num_proc),
        remove_columns=base.column_names, # drop variable-length columns
        desc=f"Packing -> {pack_len}",
    )

    if out_cap and out_cap > 0:
        packed = packed.select(range(min(out_cap, len(packed))))

    # Add constant length column (cheaper than summing masks)
    packed = packed.map(
        lambda batch: {"length": [pack_len] * len(batch["input_ids"])},
        batched=True,
        desc="Adding length column",
        num_proc=24,
        batch_size=1000,
    )

    # Tensors on CPU; Trainer moves to device efficiently
    packed = packed.with_format(type="torch", columns=["input_ids", "attention_mask", "length"])
    return packed


# ------------------------- Training script -------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="Path to converted encoder (e.g., ./gemma3-270m-encoder-bidir)")
    ap.add_argument("--dataset-path", required=True, help="Path to pretokenized dataset (save_to_disk dir)")
    ap.add_argument("--output-dir", required=True, help="Where to save checkpoints")

    # Efficiency knobs
    ap.add_argument("--pack-seq-len", type=int, default=2048, help="Packed block length (tokens)")
    ap.add_argument("--pack-cap", type=int, default=0, help="Cap the #packed blocks for quick runs (0 = no cap)")
    ap.add_argument("--pack-workers", type=int, default=max(4, (os.cpu_count() or 8)//2), help="Packing map num_proc")
    ap.add_argument("--group-by-length", action="store_true", help="(Minor) benefit if not perfectly packed")
    ap.add_argument("--sliding-window", type=int, default=128, help="Local attention window for Gemma3 (smaller => less memory)")
    ap.add_argument("--flash-attn2", action="store_true", help="Prefer FlashAttention-2 kernels")
    ap.add_argument("--gradient-checkpointing", action="store_true", help="Enable gradient checkpointing (use_reentrant=False)")
    ap.add_argument("--compile", action="store_true", help="torch.compile('max-autotune') for extra speed")
    ap.add_argument("--tokens-per-device-step", type=int, default=0,
                    help="If >0, derive per-device batch size as tokens_per_device_step // pack_seq_len")
    ap.add_argument("--dataloader-workers", type=int, default=max(4, (os.cpu_count() or 8)//2),
                    help="PyTorch DataLoader workers per process")
    ap.add_argument("--dataloader-persistent-workers", action="store_true", help="Keep workers alive between epochs")

    # Optional DeepSpeed (multi-GPU / memory scaling)
    ap.add_argument("--deepspeed-zero3", action="store_true", help="Use DeepSpeed ZeRO-3")
    ap.add_argument("--zero3-offload", action="store_true", help="ZeRO-3 CPU offload (params+optimizer)")

    # Standard training knobs
    ap.add_argument("--batch-size", type=int, default=8, help="Per-device microbatch size (overridden by tokens-per-device-step)")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--mlm-prob", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--save-steps", type=int, default=2000)
    ap.add_argument("--logging-steps", type=int, default=100)
    ap.add_argument("--pad-to-multiple-of", type=int, default=0, help="Not used for packed blocks; kept for compat")
    ap.add_argument("--max-train-pct", type=float, default=1.0, help="Use first pct of packed dataset (0<..<=1.0)")
    ap.add_argument("--run_name", type=str, help="Run name for logging")
    return ap.parse_args()


def _enable_fast_kernels():
    # Prefer Flash/SDPA kernels where possible
    try:
        from torch.backends.cuda import sdp_kernel
        sdp_kernel.enable_flash_sdp(True)
        sdp_kernel.enable_mem_efficient_sdp(True)
        sdp_kernel.enable_math_sdp(False)
    except Exception:
        pass
    # Ampere+ TF32 speedups for matmuls/convs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    # cuDNN heuristics (helpful even though seq length is fixed)
    torch.backends.cudnn.benchmark = True


def _maybe_build_deepspeed_config(args, out_dir) -> Optional[str]:
    if not args.deepspeed_zero3:
        return None
    cfg = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "zero_optimization": {
            "stage": 3,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": 5e8,
            "stage3_param_persistence_threshold": 1e6,
            "offload_param": {"device": "cpu" if args.zero3_offload else "none"},
            "offload_optimizer": {"device": "cpu" if args.zero3_offload else "none"},
        },
        "bf16": {"enabled": args.bf16},
        "fp16": {"enabled": args.fp16},
    }
    path = os.path.join(out_dir, "deepspeed_zero3.json")
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    return path


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    # ---- Prefer fast kernels on CUDA ----
    if torch.cuda.is_available():
        _enable_fast_kernels()

    # ---- Load tokenizer ----
    tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    if tok.mask_token is None:
        tok.add_special_tokens({"mask_token": "<mask>"})

    # ---- dtype selection ----
    dtype = torch.float32
    if torch.cuda.is_available():
        if args.bf16 and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        elif args.fp16:
            dtype = torch.float16

    # ---- Load model ----
    model = Gemma3EncoderForMaskedLM.from_pretrained(args.model_dir, torch_dtype=dtype)
    # Resize embeddings if we just added a mask token above
    if len(tok) != model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tok))

    # ModernBERT-like: shrink local window if available (memory saver)
    if hasattr(model.config, "sliding_window"):
        model.config.sliding_window = int(args.sliding_window)

    # Prefer FlashAttention-2 if requested
    if args.flash_attn2:
        for key in ["_attn_implementation", "attn_implementation"]:
            if hasattr(model.config, key):
                setattr(model.config, key, "flash_attention_2")

    # Encoders never use KV cache
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    # Gradient checkpointing (use_reentrant=False improves speed in many setups)
    if args.gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()

    # torch.compile (PyTorch 2.3+)
    if args.compile:
        try:
            model = torch.compile(model, mode="max-autotune")
        except Exception as e:
            print(f"[WARN] torch.compile failed: {e}")

    # ---- Build packed dataset ----
    print(f"[INFO] Packing dataset to fixed length = {args.pack_seq_len}")
    packed = make_packed_dataset(
        dataset_path=args.dataset_path,
        tokenizer=tok,
        pack_len=args.pack_seq_len,
        out_cap=(args.pack_cap if args.pack_cap > 0 else None),
        num_proc=args.pack_workers,
    )

    # (Optional) Use only a prefix of the packed dataset
    if args.max_train_pct < 1.0:
        n = len(packed)
        m = max(1, int(n * args.max_train_pct))
        packed = packed.select(range(m))
        print(f"[INFO] Subselected {m}/{n} packed blocks ({100.0*args.max_train_pct:.1f}%).")

    # ---- Collator: dynamic BERT-style masking on packed blocks ----
    # NOTE: blocks are already same length; no need for pad_to_multiple_of
    collator = DataCollatorForLanguageModeling(
        tokenizer=tok,
        mlm=True,
        mlm_probability=args.mlm_prob,
        pad_to_multiple_of=None,
    )

    # ---- Derive batch size from tokens-per-device-step if requested ----
    if args.tokens_per_device_step and args.tokens_per_device_step > 0:
        derived_bs = max(1, args.tokens_per_device_step // args.pack_seq_len)
        if derived_bs != args.batch_size:
            print(f"[INFO] Deriving per-device batch size from tokens-per-device-step:"
                  f" {args.tokens_per_device_step} // {args.pack_seq_len} -> {derived_bs}")
            args.batch_size = derived_bs

    # Throughput sanity message
    eff_tokens_per_device_step = args.batch_size * args.pack_seq_len * max(1, args.grad_accum)
    print(f"[INFO] Effective tokens per device per optimizer step ~= {eff_tokens_per_device_step:,}")

    # ---- Training args ----
    use_bf16 = (dtype == torch.bfloat16)
    use_fp16 = (dtype == torch.float16)

    ds_config_path = _maybe_build_deepspeed_config(args, args.output_dir)

    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        # save_total_limit=3,
        run_name=args.run_name, #"gemma3-mlm-train-{}-{}sql".format(str(args.sliding_window), str(args.pack_seq_len)),
        dataloader_pin_memory=True,
        dataloader_num_workers=max(1, args.dataloader_workers),
        dataloader_persistent_workers=args.dataloader_persistent_workers,
        dataloader_drop_last=True,   # avoid tiny last batch -> better kernel tuning
        report_to=['wandb'],                # no external logging overhead by default
        remove_unused_columns=False, # keep custom fields
        fp16=use_fp16,
        bf16=use_bf16,
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        group_by_length=args.group_by_length,
        length_column_name="length",
        ddp_find_unused_parameters=False,  # faster DDP
        deepspeed=ds_config_path,          # None if not using DeepSpeed
    )

    trainer = MLMAccuracyTrainer(
        model=model,
        args=targs,
        train_dataset=packed,
        data_collator=collator,
        tokenizer=tok,
    )

    print("[INFO] Starting training...")
    trainer.train()
    print("[INFO] Training complete.")

    # Save artifacts (both wrapper and raw encoder dir for clarity)
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)
    enc_dir = os.path.join(args.output_dir, "encoder")
    os.makedirs(enc_dir, exist_ok=True)
    model.save_pretrained(enc_dir)
    tok.save_pretrained(enc_dir)
    print(f"[INFO] Saved encoder to: {enc_dir}")


if __name__ == "__main__":
    main()


"""
# stick to one GPU
export CUDA_VISIBLE_DEVICES=0
# small input-pipeline win; avoids thread contention in tokenizers
export TOKENIZERS_PARALLELISM=false
# (optional) slightly better allocator behavior under load
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python train_gemma3_mlm.py \
  --model-dir ./gemma3-270m-encoder-bidir \
  --dataset-path ./data/ecom_prepared \
  --output-dir ./out_5090 \
  --pack-seq-len 2048 \
  --tokens-per-device-step 131072 \
  --flash-attn2 --bf16 --compile --gradient-checkpointing \
  --sliding-window 128 \
  --dataloader-workers $(python -c "import os; print(max(8, (os.cpu_count() or 16)//2))") \
  --dataloader-persistent-workers \
  --grad-accum 4 \
  --epochs 2 \
  --lr 2e-5 \
  --warmup-ratio 0.03 \
  --weight-decay 0.01 \
  --logging-steps 200 \
  --save-steps 5000

For multi-GPU
torchrun --nproc_per_node=8 train_gemma3_mlm.py \
  --model-dir ./gemma3-270m-encoder-bidir \
  --dataset-path ./data/ecom_prepared \
  --output-dir ./out \
  --pack-seq-len 2048 \
  --tokens-per-device-step 131072 \
  --flash-attn2 --bf16 --compile --gradient-checkpointing \
  --dataloader-workers $(python -c "import os; print(max(4, os.cpu_count()//2))") \
  --logging-steps 200

"""