"""
Time-VLM 复现 —— 汇报绘图脚本（5090 跑完后一键出图）。

在测试集上绘制时序预测曲线：真值(GroundTruth) 与 各模型预测 叠加，
以及各实验 MSE/MAE 对比柱状图。读取 ./results/{setting}/{true,pred,metrics}.npy。

用法（在仓库根目录）：
    python repro/plot_results.py                 # 用默认 spec（见下方 DEFAULT_SPEC）
    python repro/plot_results.py --spec repro/plot_spec.json   # 自定义 spec

默认 spec 中的 setting 串 = run.py 生成的 setting 名（见 result_long_term_forecast.txt）。
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_ROOT)

# 默认：h96 对比（实际 setting 串见 result_long_term_forecast.txt；predict 通道取 OT=最后一列）
# 注意：仓库 run.py 生成的 setting 中 TimeVLM 用 'clip'（小写）、PatchTST 用 'CLIP'（大写）。
DEFAULT_SPEC = [
    {"label": "GroundTruth", "setting": None, "color": "black", "kind": "gt"},
    {"label": "Time-VLM", "setting": "long_term_forecast_clip_ETTh1_512_96_TimeVLM_ETTh1_ftM_sl512_ll48_pl96_dm32_fs1.0_0", "color": "tab:blue"},
    {"label": "PatchTST", "setting": "long_term_forecast_CLIP_ETTh1_512_96_PatchTST_ETTh1_ftM_sl512_ll48_pl96_dm128_fs1.0_0", "color": "tab:orange"},
]


def load_arrays(setting):
    d = os.path.join(REPO_ROOT, "results", setting)
    return (np.load(os.path.join(d, "true.npy")),
            np.load(os.path.join(d, "pred.npy")),
            np.load(os.path.join(d, "metrics.npy")))


def plot_curve(spec, channel=-1, sample_idx=0, out_dir="repro/figures"):
    os.makedirs(out_dir, exist_ok=True)
    pred_len = None
    lines = []
    for entry in spec:
        if entry.get("kind") == "gt":
            continue
        true, pred, metrics = load_arrays(entry["setting"])
        pred_len = pred.shape[1]
        # 取 OT 通道（最后一维）
        gt = true[sample_idx, :, channel]
        pr = pred[sample_idx, :, channel]
        lines.append((entry["label"], entry.get("color", "tab:blue"), gt, pr))

    # ground truth 用第一条非 gt 条目的 true（gt 条目 setting 为 None，不可 load）
    first_model = next(e for e in spec if e.get("kind") != "gt")
    true0, _, _ = load_arrays(first_model["setting"])
    gt = true0[sample_idx, :, channel]

    plt.figure(figsize=(10, 4.5))
    t = np.arange(len(gt))
    plt.plot(t, gt, color="black", linewidth=2, label="GroundTruth")
    for label, color, _, pr in lines:
        plt.plot(t, pr, color=color, linewidth=1.5, linestyle="--", label=label)
    plt.xlabel("Horizon step")
    plt.ylabel("OT (normalized)" if pred_len else "value")
    plt.title(f"ETTh1 prediction (sample #{sample_idx}, pred_len={pred_len}, channel=OT)")
    plt.legend()
    plt.tight_layout()
    out = os.path.join(out_dir, f"curve_predlen{pred_len}.png")
    plt.savefig(out, dpi=200)
    plt.close()
    print("[saved]", out)


def plot_metric_bar(spec, out_dir="repro/figures"):
    os.makedirs(out_dir, exist_ok=True)
    labels, mses, maes = [], [], []
    pred_len = None
    for entry in spec:
        if entry.get("kind") == "gt":
            continue
        true, pred, metrics = load_arrays(entry["setting"])
        pred_len = pred.shape[1]
        # metrics.npy = [mae, mse, rmse, mape, mspe]
        labels.append(entry["label"])
        mses.append(metrics[1])
        maes.append(metrics[0])
    x = np.arange(len(labels))
    w = 0.38
    plt.figure(figsize=(6, 4))
    plt.bar(x - w / 2, mses, w, label="MSE")
    plt.bar(x + w / 2, maes, w, label="MAE")
    plt.xticks(x, labels, rotation=20)
    plt.ylabel("error (normalized)")
    plt.title(f"ETTh1 test MSE / MAE (pred_len={pred_len})")
    plt.legend()
    plt.tight_layout()
    out = os.path.join(out_dir, f"metrics_bar_predlen{pred_len}.png")
    plt.savefig(out, dpi=200)
    plt.close()
    print("[saved]", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", type=str, default=None)
    ap.add_argument("--channel", type=int, default=-1)
    ap.add_argument("--sample", type=int, default=0)
    args = ap.parse_args()

    spec = DEFAULT_SPEC
    if args.spec:
        with open(args.spec, encoding="utf-8") as f:
            spec = json.load(f)

    plot_curve(spec, channel=args.channel, sample_idx=args.sample)
    plot_metric_bar(spec)


if __name__ == "__main__":
    main()
