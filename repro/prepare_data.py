"""
ETTh1 数据集预处理 / 统计 / 校验脚本（复刻 Dataset_ETT_hour.__read_data__ 逻辑）。

输出：
  1) 数据 shape、列、train/val/test 划分、每特征 mean/std
  2) 归一化后的数据与 scaler 存盘，供直接拷贝到 RTX5090 使用
  3) 校验 CSV 完整性（行数 + MD5）

用法（在仓库根目录）：
    python repro/prepare_data.py
"""
import hashlib
import json
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(REPO_ROOT, "dataset", "ETTh1.csv")
OUT_DIR = os.path.join(REPO_ROOT, "repro", "data")
os.makedirs(OUT_DIR, exist_ok=True)

SEQ_LEN, LABEL_LEN, PRED_LEN = 512, 48, 96
# 与 data_loader.py 一致的固定时间边界（12/4/4 个月，每月 30 天 × 24 小时）
BORDER1S = [0, 12 * 30 * 24 - SEQ_LEN, 12 * 30 * 24 + 4 * 30 * 24 - SEQ_LEN]
BORDER2S = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    df_raw = pd.read_csv(CSV)
    cols = list(df_raw.columns)
    n_rows = len(df_raw)
    print("=" * 60)
    print("ETTh1.csv 原始信息")
    print(f"  路径: {CSV}")
    print(f"  行数(含表头): {n_rows + 1}  数据行数: {n_rows}")
    print(f"  列 ({len(cols)}): {cols}")
    print(f"  MD5: {md5(CSV)}")
    print(f"  大小: {os.path.getsize(CSV)} bytes")

    # features='M' -> 取除 date 外的所有列（7 通道）
    df_data = df_raw[cols[1:]]
    print(f"  特征矩阵 shape: {df_data.shape}  (7 通道: 6 电力负载特征 + OT)")

    # 时间边界
    print("=" * 60)
    print("train/val/test 划分（12/4/4 个月，固定时间边界）")
    names = ["train", "val", "test"]
    for i, name in enumerate(names):
        b1, b2 = BORDER1S[i], BORDER2S[i]
        # __len__ = len(data_x) - seq_len - pred_len + 1
        n_windows = (b2 - b1) - SEQ_LEN - PRED_LEN + 1
        print(f"  {name:5s}: rows [{b1}, {b2})  长度={b2-b1}  可生成窗口数={n_windows}")
    print(f"  注：仅使用前 {BORDER2S[2]} 行（20*30*24），末尾 {n_rows - BORDER2S[2]} 行被忽略")

    # 归一化（StandardScaler，仅在训练段 fit）
    print("=" * 60)
    print("归一化：sklearn StandardScaler（仅在训练段 [0,8640) 上 fit）")
    scaler = StandardScaler()
    scaler.fit(df_data.values[:BORDER2S[0]])
    print("  每特征 mean:", np.round(scaler.mean_, 4))
    print("  每特征 std :", np.round(scaler.scale_, 4))
    data = scaler.transform(df_data.values)
    print(f"  归一化后 shape: {data.shape}, dtype: {data.dtype}")

    # 存盘
    np.save(os.path.join(OUT_DIR, "ETTh1_normalized.npy"), data.astype(np.float32))
    with open(os.path.join(OUT_DIR, "ETTh1_scaler.json"), "w") as f:
        json.dump({"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}, f, indent=2)
    print(f"  [saved] {os.path.join(OUT_DIR, 'ETTh1_normalized.npy')}")
    print(f"  [saved] {os.path.join(OUT_DIR, 'ETTh1_scaler.json')}")

    # 抽样校验：一条 batch=1 窗口的 shape（复刻 __getitem__）
    print("=" * 60)
    s_begin, s_end = 0, SEQ_LEN
    r_begin = s_end - LABEL_LEN
    r_end = r_begin + LABEL_LEN + PRED_LEN
    seq_x = data[s_begin:s_end]
    seq_y = data[r_begin:r_end]
    print(f"  单窗口 seq_x shape: {seq_x.shape}  (期望 [{SEQ_LEN}, 7])")
    print(f"  单窗口 seq_y shape: {seq_y.shape}  (期望 [{LABEL_LEN + PRED_LEN}, 7])")
    print("=" * 60)
    print("预处理完成。拷贝到 5090 仅需 dataset/ETTh1.csv + dataset/prompt_bank/")


if __name__ == "__main__":
    main()
