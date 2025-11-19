#!/usr/bin/env python
# tokenize_ds.py
"""
Pre-tokenize Ecom‑niverse for Gemma‑3 (bi-encoder) training.

- No padding at preproc time (the trainer will pack).
- Preserve long texts via return_overflowing_tokens (+stride).
- Optional language filter.
- Saves tokenizer (with <mask> if added) alongside the dataset.
"""

import argparse, logging, os
from typing import Optional
import datasets
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

def _resolve_dataset(
    dataset_name: Optional[str],
    dataset_path: Optional[str],
    split: str,
    seed: int,
    max_samples: Optional[int],
    lang: Optional[str],
    num_proc: int,
) -> Dataset:
    if dataset_path:
        logger.info("Loading dataset from disk: %s", dataset_path)
        ds = datasets.load_from_disk(dataset_path)
        if isinstance(ds, DatasetDict):
            ds = ds.get(split) if split in ds else list(ds.values())[0]
    elif dataset_name:
        logger.info("Loading dataset from hub: %s (%s)", dataset_name, split)
        # Use non-streaming Dataset so we can shuffle/filter/map efficiently and save to disk
        ds = datasets.load_dataset(dataset_name, split=split)
    else:
        raise ValueError("Provide either --dataset-name or --dataset-path")

    # Optional language filter if column exists
    if lang and "lang" in ds.column_names:
        logger.info("Filtering to lang == %s", lang)
        ds = ds.filter(lambda ex: ex.get("lang", None) == lang, num_proc=num_proc)

    # Remove empty / whitespace-only
    if "text" in ds.column_names:
        ds = ds.filter(lambda ex: isinstance(ex["text"], str) and ex["text"].strip() != "", num_proc=num_proc)
    else:
        raise ValueError("Dataset must have a 'text' column")

    # Shuffle before selecting
    ds = ds.shuffle(seed=seed)
    if max_samples and max_samples > 0:
        logger.info("Selecting first %d samples after shuffle", max_samples)
        ds = ds.select(range(min(max_samples, len(ds))))
    logger.info("Base dataset size: %d", len(ds))
    return ds


def build_dataset(
    tokenizer: AutoTokenizer,
    ds: Dataset,
    max_length: int,
    min_length: int,
    stride: int,
    num_proc: int,
    return_overflowing_tokens: bool = True,
) -> Dataset:
    """
    Tokenize without padding; split long examples into multiple windows using
    `return_overflowing_tokens=True` to avoid discarding tokens.
    """

    def tokenize_fn(batch):
        out = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=False,                    # keep variable length
            return_attention_mask=True,
            return_overflowing_tokens=return_overflowing_tokens,   # keep all content
            stride=stride,                    # small overlap is helpful
        )
        # `datasets` will flatten lists of lists appropriately
        return { "input_ids": out["input_ids"], "attention_mask": out["attention_mask"] }

    logger.info("Tokenizing (max_length=%d, stride=%d) with num_proc=%d", max_length, stride, num_proc)
    logger.info("Return overflowing tokens: %s", return_overflowing_tokens)
    tokenized = ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=[c for c in ds.column_names if c != "text"],  # drop others; tokenizer doesn't need them
        num_proc=num_proc,
        desc="Tokenizing",
        batch_size=1024,
    )

    # Drop super-short sequences (little training value)
    tokenized = tokenized.filter(lambda ex: len(ex["input_ids"]) >= min_length, num_proc=num_proc)
    tokenized = tokenized.remove_columns(["text"])
    tokenized = tokenized.with_format(type="python")  # Arrow on disk; Trainer will set torch format later

    logger.info("Finished tokenizing %d sequences", len(tokenized))
    return tokenized


def main():
    parser = argparse.ArgumentParser(description="Prepare Ecom‑niverse for Gemma‑3 bi-encoder MLM")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset-name", type=str, help="HF hub dataset name (e.g., thebajajra/Ecom-niverse)")
    src.add_argument("--dataset-path", type=str, help="Local dataset dir created via datasets.save_to_disk")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output-dir", type=str, required=True)

    # Tokenizer (IMPORTANT: use the same one you train with)
    parser.add_argument("--tokenizer", type=str, default="google/gemma-3-270m",
                        help="Path or name. Prefer your converted encoder tokenizer dir for perfect match.")
    parser.add_argument("--add-mask-token", action="store_true",
                        help="Add <mask> if missing and save tokenizer next to dataset")

    # Sampling / filtering
    parser.add_argument("--max-samples", type=int, default=0, help="0 = use all")
    parser.add_argument("--lang", type=str, default="en", help="Language code to keep (if dataset has 'lang')")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-proc", type=int, default=min(32, os.cpu_count() or 8))

    # Tokenization windowing
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--min-length", type=int, default=16)
    parser.add_argument("--stride", type=int, default=64, help="Token overlap when splitting long texts")
    parser.add_argument("--return-overflowing-tokens", action="store_true",
                        help="Keep all tokens by splitting long texts into multiple windows")
    

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Reduce thread oversubscription (we parallelize with processes)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if args.add_mask_token and tok.mask_token is None:
        logger.info("Adding <mask> token to tokenizer")
        tok.add_special_tokens({"mask_token": "<mask>"})

    # Resolve + tokenize
    ds = _resolve_dataset(
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        split=args.split,
        seed=args.seed,
        max_samples=None if args.max_samples == 0 else args.max_samples,
        lang=args.lang if args.lang else None,
        num_proc=args.num_proc,
    )
    tokenized = build_dataset(
        tokenizer=tok,
        ds=ds,
        max_length=args.max_length,
        min_length=args.min_length,
        stride=args.stride,
        num_proc=args.num_proc,
        return_overflowing_tokens=args.return_overflowing_tokens,
    )

    # Save artifacts
    logger.info("Saving processed dataset to %s", args.output_dir)
    tokenized.save_to_disk(args.output_dir)
    # Save tokenizer too (so training can load the exact same one)
    tok_dir = os.path.join(args.output_dir, "tokenizer")
    tok.save_pretrained(tok_dir)
    logger.info("Saved tokenizer to %s", tok_dir)
    logger.info("Done. Wrote %d sequences to %s", len(tokenized), args.output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()


"""
python tokenize_ds.py \
  --dataset-path ecomniverse_sampled.hf \
  --output-dir data/ecom_prepared \
  --tokenizer ./gemma3-270m-encoder-bidir \
  --add-mask-token \
  --max-samples 2000000 \
  --max-length 2048 --stride 64 \
  --num-proc 24

python tokenize_ds.py \
  --dataset-name thebajajra/Ecomniverse-sampled \
  --output-dir data/ecom_sampled_prepared \
  --tokenizer ./gemma3-270m-encoder-bidir-model \
  --add-mask-token \
  --max-samples 100000000 \
  --max-length 2048 \
  --stride 64 \
  --num-proc 24 \
  --return-overflowing-tokens


"""