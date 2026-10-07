#!/usr/bin/env bash
# ============================================================
# Time-VLM 复现 —— RTX5090 32GB 一键训练脚本（Linux）
# 前置：已激活 timevlm 环境、已下载 ./dataset/ETTh1.csv、CLIP 权重可下载。
#
# 用法（在仓库根目录执行）：
#   bash repro/run_5090.sh                # 顺序跑全部 5 组（full h96/h720、PatchTST h96/h720、w/o-VLM h720）
#   bash repro/run_5090.sh --skip-ablation # 跳过 w/o-VLM 消融
#
# 注意：本脚本会真正训练（在 5090 上）。本地 RTX3050 4GB 禁止运行。
# ============================================================
set -e
cd "$(dirname "$0")/.."   # 切到仓库根目录

export TOKENIZERS_PARALLELISM=false
# 若无法访问 huggingface.co（国内网络），取消下一行注释：
# export HF_ENDPOINT=https://hf-mirror.com

GPU="${GPU:-0}"

echo ">>> [1/5] Time-VLM full, horizon=96 (d_model=32, mem_gate=False, dropout=0.1)"
python -u run.py --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/ --data_path ETTh1.csv --model_id ETTh1_512_96 \
  --model TimeVLM --data ETTh1 --features M --seq_len 512 --label_len 48 \
  --pred_len 96 --d_model 32 --e_layers 2 --d_layers 1 --factor 3 \
  --enc_in 7 --dec_in 7 --c_out 7 --des Exp --itr 1 --gpu $GPU --use_amp \
  --train_epochs 15 --image_size 56 --norm_const 0.4 --periodicity 24 \
  --three_channel_image True --finetune_vlm False --batch_size 32 \
  --learning_rate 0.001 --num_workers 32 --vlm_type clip \
  --use_mem_gate False --dropout 0.1 --percent 1.0

echo ">>> [2/5] Time-VLM full, horizon=720 (d_model=256, mem_gate=True, dropout=0.3)"
python -u run.py --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/ --data_path ETTh1.csv --model_id ETTh1_512_720 \
  --model TimeVLM --data ETTh1 --features M --seq_len 512 --label_len 48 \
  --pred_len 720 --d_model 256 --e_layers 2 --d_layers 1 --factor 3 \
  --enc_in 7 --dec_in 7 --c_out 7 --des Exp --itr 1 --gpu $GPU --use_amp \
  --train_epochs 15 --image_size 56 --norm_const 0.4 --periodicity 24 \
  --three_channel_image True --finetune_vlm False --batch_size 32 \
  --learning_rate 0.001 --num_workers 32 --vlm_type clip \
  --use_mem_gate True --dropout 0.3 --percent 1.0

echo ">>> [3/5] PatchTST baseline, horizon=96"
python -u run.py --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/ --data_path ETTh1.csv --model_id ETTh1_512_96 \
  --model PatchTST --data ETTh1 --features M --seq_len 512 --label_len 48 \
  --pred_len 96 --d_model 128 --n_heads 8 --e_layers 2 --d_ff 768 \
  --dropout 0.1 --activation gelu --factor 3 --enc_in 7 --dec_in 7 --c_out 7 \
  --des Exp --itr 1 --gpu $GPU --use_amp --train_epochs 15 --batch_size 32 \
  --learning_rate 0.001 --num_workers 32 --percent 1.0

echo ">>> [4/5] PatchTST baseline, horizon=720"
python -u run.py --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/ --data_path ETTh1.csv --model_id ETTh1_512_720 \
  --model PatchTST --data ETTh1 --features M --seq_len 512 --label_len 48 \
  --pred_len 720 --d_model 128 --n_heads 8 --e_layers 2 --d_ff 768 \
  --dropout 0.1 --activation gelu --factor 3 --enc_in 7 --dec_in 7 --c_out 7 \
  --des Exp --itr 1 --gpu $GPU --use_amp --train_epochs 15 --batch_size 32 \
  --learning_rate 0.001 --num_workers 32 --percent 1.0

if [ "$1" != "--skip-ablation" ]; then
  echo ">>> [5/5] Time-VLM w/o VLM ablation, horizon=720（先打补丁）"
  python repro/patch_w_out_vlm.py
  python -u run.py --task_name long_term_forecast --is_training 1 \
    --root_path ./dataset/ --data_path ETTh1.csv --model_id ETTh1_512_720_wovlm \
    --model TimeVLM --data ETTh1 --features M --seq_len 512 --label_len 48 \
    --pred_len 720 --d_model 256 --e_layers 2 --d_layers 1 --factor 3 \
    --enc_in 7 --dec_in 7 --c_out 7 --des Exp --itr 1 --gpu $GPU --use_amp \
    --train_epochs 15 --image_size 56 --norm_const 0.4 --periodicity 24 \
    --three_channel_image True --finetune_vlm False --batch_size 32 \
    --learning_rate 0.001 --num_workers 32 --vlm_type clip \
    --use_mem_gate True --dropout 0.3 --percent 1.0 \
    --w_out_visual True --w_out_text True
  python repro/patch_w_out_vlm.py --revert
fi

echo ">>> 全部训练完成。结果见 result_long_term_forecast.txt 与 ./results/{setting}/"
echo ">>> 一键出图/出表： python repro/plot_results.py && python repro/make_table.py"
