"""
Time-VLM 复现 —— 单样本前向测试（FORWARD-ONLY，严格禁止训练）。

硬性约束：本脚本只做 forward 前向推理，绝不调用 loss.backward()、绝不创建 optimizer、
绝不更新参数、绝不执行任何 epoch 训练循环。仅用于验证：
  1) 模型能否正常导入/实例化
  2) 输入/输出张量维度是否符合论文描述
  3) 是否 OOM（若 OOM，记录发生阶段与显存占用）

用法（在仓库根目录执行，需先激活 timevlm 环境）：
    python repro/forward_test.py                       # 默认：TimeVLM h96 + h720 + PatchTST h96 + h720（forward-only）
    python repro/forward_test.py --model TimeVLM --pred_len 720
    python repro/forward_test.py --with-grad           # 额外做一次 grad-enabled forward（测量训练态显存，仍不 backward）

CLIP 权重下载说明：若本地无缓存且无法访问 huggingface.co，请设置环境变量
    export HF_ENDPOINT=https://hf-mirror.com
"""
import argparse
import json
import os
import random
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch

SEED = 2024


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # 论文仓库未设置此项，这里为确定性补齐


def device_info():
    if not torch.cuda.is_available():
        return {"cuda": False, "note": "CUDA 不可用"}
    p = torch.cuda.get_device_properties(0)
    return {
        "cuda": True,
        "name": torch.cuda.get_device_name(0),
        "total_memory_MB": round(p.total_memory / 1e6, 1),
        "cuda_capability": f"{p.major}.{p.minor}",
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
    }


def mem_snapshot(tag=""):
    if not torch.cuda.is_available():
        return None
    return {
        "tag": tag,
        "allocated_MB": round(torch.cuda.memory_allocated() / 1e6, 1),
        "peak_MB": round(torch.cuda.max_memory_allocated() / 1e6, 1),
        "reserved_MB": round(torch.cuda.memory_reserved() / 1e6, 1),
    }


def make_timevlm_config(pred_len, d_model, use_mem_gate, dropout):
    from types import SimpleNamespace
    with open(os.path.join(REPO_ROOT, "dataset", "prompt_bank", "ETT.txt"), encoding="utf-8") as f:
        content = f.read()
    return SimpleNamespace(
        gpu=0, use_gpu=True, use_mem_gate=use_mem_gate,
        patch_memory_size=100, patch_len=16, d_model=d_model, stride=8, padding=8,
        dropout=dropout, seq_len=512, pred_len=pred_len, top_k=5,
        three_channel_image=True, image_size=56, periodicity=24, learnable_image=True,
        norm_const=0.4, finetune_vlm=False, vlm_type="clip", content=content,
        save_images=False, w_out_query=False, memory_bank_size=20,
        use_cross_attention=True, w_out_visual=False, w_out_text=False,
    )


def make_patchtst_config(pred_len):
    from types import SimpleNamespace
    return SimpleNamespace(
        task_name="long_term_forecast", seq_len=512, pred_len=pred_len,
        d_model=128, d_ff=768, n_heads=8, e_layers=2, d_layers=1,
        dropout=0.1, activation="gelu", factor=3, enc_in=7, dec_in=7, c_out=7,
    )


def load_etth1_batch(seq_len=512, label_len=48, pred_len=96, batch_size=1, idx=0):
    """复刻 Dataset_ETT_hour 的读取 + StandardScaler(仅训练段 fit) + 滑窗逻辑，返回一条 batch。"""
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    df = pd.read_csv(os.path.join(REPO_ROOT, "dataset", "ETTh1.csv"))
    data = df[df.columns[1:]].values.astype(np.float32)  # 7 通道
    scaler = StandardScaler()
    scaler.fit(data[:8640])  # 仅在训练段 fit（与 data_loader 一致）
    data = scaler.transform(data)
    s_begin, s_end = idx, idx + seq_len
    r_begin = s_end - label_len
    r_end = r_begin + label_len + pred_len
    seq_x = data[s_begin:s_end]                 # [seq_len, 7]
    seq_y = data[r_begin:r_end]                 # [label_len+pred_len, 7]
    x = torch.from_numpy(seq_x).unsqueeze(0).float()   # [1, seq_len, 7]
    y = torch.from_numpy(seq_y).unsqueeze(0).float()
    return x, y, scaler


def run_timevlm(pred_len, d_model, use_mem_gate, dropout, with_grad=False):
    r = {"model": "TimeVLM", "pred_len": pred_len, "d_model": d_model,
         "use_mem_gate": use_mem_gate, "dropout": dropout, "with_grad": with_grad}
    # --- 阶段1：导入 ---
    try:
        from src.TimeVLM.model import Model
        r["stage_import"] = "ok"
    except Exception as e:
        r["stage_import"] = f"FAILED {type(e).__name__}: {str(e)[:500]}"
        r["status"] = "FAILED"
        return r
    # --- 阶段2：实例化 ---
    try:
        cfg = make_timevlm_config(pred_len, d_model, use_mem_gate, dropout)
        model = Model(cfg)
        r["stage_init"] = "ok"
        r["params_total"] = sum(p.numel() for p in model.parameters())
        r["params_trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    except Exception as e:
        r["stage_init"] = f"FAILED {type(e).__name__}: {str(e)[:800]}"
        r["status"] = "FAILED"
        return r
    # --- 阶段3：前向 ---
    try:
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        model = model.to(device)
        model.eval()
        x, _, _ = load_etth1_batch(seq_len=512, label_len=48, pred_len=pred_len)
        x = x.to(device)
        r["input_shape"] = list(x.shape)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        ctx = torch.enable_grad() if with_grad else torch.no_grad()
        with ctx:
            y = model(x)
        r["output_shape"] = list(y.shape)
        r["expected_shape"] = [1, pred_len, 7]
        r["dim_match"] = list(y.shape) == [1, pred_len, 7]
        r["output_finite"] = bool(torch.isfinite(y).all().item())
        r["stage_forward"] = "ok"
        r["mem"] = mem_snapshot("post-forward")
        r["status"] = "PASS" if r["dim_match"] and r["output_finite"] else "DIM/NUM-ISSUE"
    except torch.cuda.OutOfMemoryError as e:
        r["stage_forward"] = "OOM"
        r["mem"] = mem_snapshot("at-OOM")
        r["oom_stage"] = "forward"
        r["error"] = str(e)[:300]
        r["status"] = "OOM"
    except Exception as e:
        r["stage_forward"] = f"ERROR {type(e).__name__}"
        r["mem"] = mem_snapshot("at-error")
        r["error"] = str(e)[:800]
        r["status"] = "ERROR"
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return r


def run_patchtst(pred_len, with_grad=False):
    r = {"model": "PatchTST", "pred_len": pred_len, "with_grad": with_grad}
    try:
        from models.PatchTST import Model
        r["stage_import"] = "ok"
    except Exception as e:
        r["stage_import"] = f"FAILED {type(e).__name__}: {str(e)[:500]}"
        r["status"] = "FAILED"
        return r
    try:
        cfg = make_patchtst_config(pred_len)
        model = Model(cfg)
        r["stage_init"] = "ok"
        r["params_total"] = sum(p.numel() for p in model.parameters())
    except Exception as e:
        r["stage_init"] = f"FAILED {type(e).__name__}: {str(e)[:800]}"
        r["status"] = "FAILED"
        return r
    try:
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        model = model.to(device)
        model.eval()
        x, _, _ = load_etth1_batch(seq_len=512, label_len=48, pred_len=pred_len)
        x = x.to(device)
        r["input_shape"] = list(x.shape)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        ctx = torch.enable_grad() if with_grad else torch.no_grad()
        with ctx:
            y = model(x, None, None, None)  # PatchTST forward(x_enc, x_mark, x_dec, x_mark)
        r["output_shape"] = list(y.shape)
        r["expected_shape"] = [1, pred_len, 7]
        r["dim_match"] = list(y.shape) == [1, pred_len, 7]
        r["output_finite"] = bool(torch.isfinite(y).all().item())
        r["stage_forward"] = "ok"
        r["mem"] = mem_snapshot("post-forward")
        r["status"] = "PASS" if r["dim_match"] and r["output_finite"] else "DIM/NUM-ISSUE"
    except torch.cuda.OutOfMemoryError as e:
        r["stage_forward"] = "OOM"
        r["mem"] = mem_snapshot("at-OOM")
        r["oom_stage"] = "forward"
        r["error"] = str(e)[:300]
        r["status"] = "OOM"
    except Exception as e:
        r["stage_forward"] = f"ERROR {type(e).__name__}"
        r["mem"] = mem_snapshot("at-error")
        r["error"] = str(e)[:800]
        r["status"] = "ERROR"
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="all", choices=["all", "TimeVLM", "PatchTST"])
    ap.add_argument("--pred_len", type=int, default=None)
    ap.add_argument("--with-grad", action="store_true")
    args = ap.parse_args()

    set_seed()
    results = []
    results.append({"type": "device", **device_info()})

    if args.model in ("all", "TimeVLM"):
        pls = [args.pred_len] if args.pred_len else [96, 720]
        cfg_map = {96: (32, False, 0.1), 720: (256, True, 0.3)}
        for pl in pls:
            dm, mg, dp = cfg_map.get(pl, (32, False, 0.1))
            results.append(run_timevlm(pl, dm, mg, dp, with_grad=args.with_grad))

    if args.model in ("all", "PatchTST"):
        pls = [args.pred_len] if args.pred_len else [96, 720]
        for pl in pls:
            results.append(run_patchtst(pl, with_grad=args.with_grad))

    print(json.dumps(results, ensure_ascii=False, indent=2))

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "forward_test_report.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
