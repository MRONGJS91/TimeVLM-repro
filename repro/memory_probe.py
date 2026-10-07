"""
Time-VLM 显存探测脚本（供 RTX5090 32GB 使用；本地 RTX3050 4GB 只允许 --forward-only）。

功能：扫描不同 batch_size × FP16(AMP) × 梯度检查点 组合下的峰值显存，识别 OOM 风险，
      为 RTX5090 给出安全 batch size 与梯度累积建议。

⚠️ 硬性约束：
  - 在 RTX3050（4GB）上必须使用 --forward-only（只 forward，不 backward），否则违反"禁止训练"红线。
  - 在 RTX5090 上才允许去掉 --forward-only（做单步 forward+backward 显存探测，属于内存测算，非 epoch 训练）。

用法：
    # 本地 3050：只测前向显存（batch 1~8，FP16 开关）
    python repro/memory_probe.py --forward-only --batch-list 1,2,4,8
    # 5090：完整前向+反向显存（batch 1~64，FP16 开关，梯度检查点开关）
    python repro/memory_probe.py --batch-list 1,2,4,8,16,32,64
"""
import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import torch

from repro.forward_test import (make_timevlm_config, make_patchtst_config,
                                load_etth1_batch, set_seed)


def build_model(kind, pred_len):
    if kind == "TimeVLM":
        from src.TimeVLM.model import Model
        cfg_map = {96: (32, False, 0.1), 720: (256, True, 0.3)}
        dm, mg, dp = cfg_map.get(pred_len, (32, False, 0.1))
        return Model(make_timevlm_config(pred_len, dm, mg, dp))
    else:
        from models.PatchTST import Model
        return Model(make_patchtst_config(pred_len))


def enable_grad_checkpointing(model):
    """对 CLIP 骨干开启 transformers 的 gradient checkpointing；TimeVLM 时序分支不支持原生 ckpt。"""
    enabled = False
    vlm = getattr(model, "vlm_model", None)
    if vlm is not None and hasattr(vlm, "gradient_checkpointing_enable"):
        try:
            vlm.gradient_checkpointing_enable()
            enabled = True
        except Exception:
            pass
    return enabled


def probe(kind, pred_len, batch_list, fp16, grad_ckpt, forward_only):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda:0")
    model = build_model(kind, pred_len).to(device)
    model.train()
    ckpt_enabled = False
    if grad_ckpt:
        ckpt_enabled = enable_grad_checkpointing(model)

    seq_len = 512
    x, y, _ = load_etth1_batch(seq_len=seq_len, label_len=48, pred_len=pred_len)
    # 构造 label（取 y 的最后 pred_len 步，匹配 exp 中 outputs[:, -pred_len:] 的监督信号）
    label = y[:, -pred_len:, :].to(device)
    criterion = torch.nn.MSELoss()

    results = []
    for bs in batch_list:
        xb = x.expand(bs, -1, -1).contiguous().to(device)
        lb = label.expand(bs, -1, -1).contiguous().to(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        rec = {"kind": kind, "pred_len": pred_len, "batch_size": bs,
               "fp16": fp16, "grad_ckpt": ckpt_enabled, "forward_only": forward_only}
        try:
            if kind == "TimeVLM":
                def fwd():
                    return model(xb)
            else:
                def fwd():
                    return model(xb, None, None, None)

            if forward_only:
                with torch.no_grad():
                    out = fwd()
            elif fp16:
                with torch.cuda.amp.autocast():
                    out = fwd()
                loss = criterion(out, lb)
                loss.backward()
            else:
                out = fwd()
                loss = criterion(out, lb)
                loss.backward()

            rec["ok"] = True
            rec["peak_MB"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)
        except torch.cuda.OutOfMemoryError:
            rec["ok"] = False
            rec["OOM"] = True
            rec["peak_MB"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)
        except Exception as e:
            rec["ok"] = False
            rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        results.append(rec)
        torch.cuda.empty_cache()
    del model
    torch.cuda.empty_cache()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", type=str, default="TimeVLM", choices=["TimeVLM", "PatchTST"])
    ap.add_argument("--pred-len", type=int, default=720)
    ap.add_argument("--batch-list", type=str, default="1,2,4,8")
    ap.add_argument("--fp16-list", type=str, default="0,1")
    ap.add_argument("--grad-ckpt-list", type=str, default="0")
    ap.add_argument("--forward-only", action="store_true",
                    help="只 forward 不 backward（本地 3050 必须开启）")
    args = ap.parse_args()

    set_seed()
    device = torch.device("cuda:0")
    total = torch.cuda.get_device_properties(0).total_memory / 1e6
    print(f"GPU={torch.cuda.get_device_name(0)} total={total:.0f}MB")

    batch_list = [int(b) for b in args.batch_list.split(",") if b.strip()]
    fp16_list = [bool(int(f)) for f in args.fp16_list.split(",") if f.strip()]
    grad_ckpt_list = [bool(int(g)) for g in args.grad_ckpt_list.split(",") if g.strip()]

    if args.forward_only:
        fp16_list = [False]  # forward-only 时 fp16 影响有限，仅测 fp32 前向
        grad_ckpt_list = [False]

    all_results = []
    for gc in grad_ckpt_list:
        for fp in fp16_list:
            all_results += probe(args.kind, args.pred_len, batch_list, fp, gc, args.forward_only)

    print(json.dumps(all_results, ensure_ascii=False, indent=2))
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory_probe_report.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"gpu_total_MB": round(total, 0), "results": all_results}, f,
                  ensure_ascii=False, indent=2)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
