# stick to one GPU
export CUDA_VISIBLE_DEVICES=0
# small input-pipeline win; avoids thread contention in tokenizers
export TOKENIZERS_PARALLELISM=false
# (optional) slightly better allocator behavior under load
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python train_gemma3_mlm.py \
  --model-dir ./gemma3-270m-encoder-bidir-model \
  --dataset-path /home/bajajra/workspace/rexgemma/data/ecom_sampled_prepared \
  --output-dir ./gemma3-270m_sw128_4096tk \
  --pack-seq-len 2048 \
  --tokens-per-device-step 8192 \
  --flash-attn2 --bf16 --gradient-checkpointing \
  --sliding-window 128 \
  --dataloader-workers $(python -c "import os; print(max(8, (os.cpu_count() or 16)//2))") \
  --dataloader-persistent-workers \
  --grad-accum 8 \
  --epochs 1 \
  --lr 1e-4 \
  --warmup-ratio 0.03 \
  --weight-decay 0.01 \
  --logging-steps 200 \
  --save-steps 5000

