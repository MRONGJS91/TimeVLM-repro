#!/usr/bin/env python3
"""
矩阵结果聚合：解析 result_long_term_forecast.txt + zero_shot_results.json，
与 paper_reference.json 对比，生成 repro/矩阵结果汇总.md。
"""
import json
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
RESULT = os.path.join(REPO, "result_long_term_forecast.txt")
REF = json.load(open(os.path.join(REPO, "repro", "paper_reference.json")))
ZERO = json.load(open(os.path.join(REPO, "repro", "zero_shot_results.json")))

DSET_ORDER = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Exchange", "ECL"]
MODELS = ["TimeVLM", "PatchTST", "iTransformer", "TimeLLM"]
HORIZONS = [96, 192, 336, 720]
SETTING_MAP = {"full": "full_shot", "few": "few_shot_10p", "zero": "zero_shot"}


def parse_setting(setting):
    """-> (model, dataset, horizon, kind)"""
    if "TimeVLM" in setting:
        model = "TimeVLM"
    elif "PatchTST" in setting:
        model = "PatchTST"
    elif "iTransformer" in setting:
        model = "iTransformer"
    elif "TimeLLM" in setting:
        model = "TimeLLM"
    else:
        model = "?"
    kind = "full"
    if setting.startswith("few_shot"):
        kind = "few"
    m = re.search(r"(ETTh[12]|ETTm[12]|Exchange|ECL)_512_(\d+)", setting)
    dataset = m.group(1) if m else "?"
    horizon = int(m.group(2)) if m else -1
    return model, dataset, horizon, kind


def load_results():
    rows = {}
    text = open(RESULT).read()
    for sm in re.finditer(r"(\S+)\s*\nmse:\s*([0-9.eE+-]+),\s*mae:\s*([0-9.eE+-]+)", text):
        setting = sm.group(1)
        if "wovlm" in setting or "e1fix" in setting:
            continue  # 消融/修复实验不进入矩阵汇总
        model, dataset, horizon, kind = parse_setting(setting)
        rows[(model, dataset, horizon, kind)] = (float(sm.group(2)), float(sm.group(3)))
    for name, v in ZERO.items():
        m2 = re.search(r"zero_(\w+)_(\w+)_(\w+)_512_(\d+)", name)
        if not m2:
            continue
        model, src, tgt, h = m2.group(1), m2.group(2), m2.group(3), int(m2.group(4))
        rows[(model, f"{src}->{tgt}", h, "zero")] = (v["mse"], v["mae"])
    return rows


def fmt(mse, mae):
    return f"{mse:.4f} / {mae:.4f}"


def rel(mse, ref):
    return (mse - ref) / ref * 100


def main():
    rows = load_results()
    lines = ["# Time-VLM 完整复现矩阵 —— 结果汇总", ""]
    lines.append(f"> 生成：矩阵 {len(rows)} 组结果。格式 MSE/MAE（归一化空间，与论文口径一致）。")
    lines.append("> 论文参考值来源：Table 16（full）/ Table 13（few 10%）/ Table 14（zero），见 paper_reference.json。")
    lines.append("")

    for kind, title in [("full", "Full-shot（论文 Table 16 对照）"),
                        ("few", "Few-shot 10%（论文 Table 13 对照）")]:
        lines.append(f"## {title}")
        lines.append("")
        for dset in DSET_ORDER:
            ref_ds = dset if dset != "ECL" else "ECL"
            has_ref = dset in REF["full_shot"]
            lines.append(f"### {dset}" + ("" if has_ref else "（论文无此数据集）"))
            lines.append("")
            lines.append("| Model | 96 | 192 | 336 | 720 | Avg |")
            lines.append("|---|---|---|---|---|---|")
            for model in MODELS:
                cells, avg_m, avg_a = [], [], []
                for h in HORIZONS:
                    key = (model, dset, h, kind)
                    if key in rows:
                        m, a = rows[key]
                        cells.append(fmt(m, a))
                        avg_m.append(m)
                        avg_a.append(a)
                    else:
                        cells.append("—")
                if not avg_m:
                    continue
                line = f"| {model} | " + " | ".join(cells)
                line += f" | {sum(avg_m)/len(avg_m):.4f} / {sum(avg_a)/len(avg_a):.4f} |"
                lines.append(line)
            if has_ref:
                lines.append("")
                lines.append("| 论文参考 | 96 | 192 | 336 | 720 | Avg |")
                lines.append("|---|---|---|---|---|---|")
                for model in ["TimeVLM", "Time-LLM", "PatchTST"]:
                    if model not in REF["full_shot"][dset]:
                        continue
                    r = REF["full_shot"][dset][model]
                    cells = [fmt(r[str(h)]["mse"], r[str(h)]["mae"]) for h in HORIZONS]
                    lines.append(f"| {model} | " + " | ".join(cells) +
                                 f" | {r['avg']['mse']:.4f} / {r['avg']['mae']:.4f} |")
                lines.append("")
                # TimeVLM 相对误差
                tv = REF["full_shot"][dset]["TimeVLM"]
                errs = []
                for h in HORIZONS:
                    key = ("TimeVLM", dset, h, kind)
                    if key in rows:
                        errs.append(rel(rows[key][0], tv[str(h)]["mse"]))
                if errs:
                    lines.append(f"**TimeVLM vs 论文相对误差**：逐 horizon " +
                                 ", ".join(f"{e:+.1f}%" for e in errs) +
                                 f"；平均 {sum(errs)/len(errs):+.1f}%")
                    lines.append("")
            lines.append("")

    lines.append("## Zero-shot 迁移（论文 Table 14 对照，ETT 配对）")
    lines.append("")
    for pair in ["ETTh1->ETTh2", "ETTh2->ETTh1", "ETTm1->ETTm2", "ETTm2->ETTm1"]:
        lines.append(f"### {pair}")
        lines.append("")
        lines.append("| Model | 96 | 192 | 336 | 720 |")
        lines.append("|---|---|---|---|---|")
        for model in ["TimeVLM", "TimeLLM"]:
            cells = []
            for h in HORIZONS:
                key = (model, pair, h, "zero")
                cells.append(fmt(*rows[key]) if key in rows else "—")
            lines.append(f"| {model} | " + " | ".join(cells) + " |")
        ref = REF["zero_shot"].get(pair.replace("->", "→"), {})
        if ref:
            lines.append(f"| 论文 TimeVLM | " + " | ".join(
                fmt(ref["TimeVLM"][str(h)]["mse"], ref["TimeVLM"][str(h)]["mae"]) for h in HORIZONS) + " |")
            lines.append(f"| 论文 Time-LLM | " + " | ".join(
                fmt(ref["Time-LLM"][str(h)]["mse"], ref["Time-LLM"][str(h)]["mae"]) for h in HORIZONS) + " |")
        lines.append("")

    lines.append("## 判定小结")
    lines.append("")
    lines.append("- Tier 1：TimeVLM MSE 相对论文 ≤5%；Tier 2：≤10%；其余为 Tier 3。")
    lines.append("- 注：Exchange 无论文参考；ECL 仅 h96/h720 精简组。")
    lines.append("")
    for dset in ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]:
        for kind in ["full", "few"]:
            errs = []
            for h in HORIZONS:
                key = ("TimeVLM", dset, h, kind)
                if key in rows and dset in REF[SETTING_MAP[kind]]:
                    refm = REF[SETTING_MAP[kind]][dset]["TimeVLM"][str(h)]["mse"]
                    errs.append(rel(rows[key][0], refm))
            if errs:
                avg = sum(errs) / len(errs)
                tier = "Tier 1" if abs(avg) <= 5 else ("Tier 2" if abs(avg) <= 10 else "Tier 3")
                lines.append(f"- TimeVLM {dset} {kind}-shot: 平均相对误差 {avg:+.1f}% → **{tier}**")
    out = os.path.join(REPO, "repro", "矩阵结果汇总.md")
    open(out, "w").write("\n".join(lines) + "\n")
    print(f"[saved] {out}  ({len(rows)} rows aggregated)")


if __name__ == "__main__":
    main()
