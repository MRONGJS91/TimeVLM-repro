"""
Time-VLM 复现 —— 指标汇总脚本（5090 跑完后一键出表）。

自动收集 result_long_term_forecast.txt 中的 MSE/MAE，生成：
  1) markdown 表格
  2) LaTeX booktabs 表格代码
并支持与论文参考值对比（可选，通过 --reference 传入）。

用法（在仓库根目录）：
    python repro/make_table.py
    python repro/make_table.py --reference repro/paper_reference.json
"""
import argparse
import json
import os
import re

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_ROOT)

RESULT_FILE = "result_long_term_forecast.txt"


def parse_results():
    """解析 result_long_term_forecast.txt -> list of {setting, mse, mae, dtw}"""
    rows = []
    if not os.path.exists(RESULT_FILE):
        print(f"[warn] 未找到 {RESULT_FILE}")
        return rows
    with open(RESULT_FILE, encoding="utf-8") as f:
        text = f.read()
    # 匹配：setting 行 + "mse: X, mae: Y, dtw: Z" 行
    pattern = re.compile(
        r"(?P<setting>\S+)\s*\nmse:\s*(?P<mse>[0-9.eE+-]+),\s*mae:\s*(?P<mae>[0-9.eE+-]+)"
    )
    for m in pattern.finditer(text):
        rows.append({
            "setting": m.group("setting"),
            "mse": float(m.group("mse")),
            "mae": float(m.group("mae")),
        })
    return rows


def parse_setting(setting):
    """从 setting 串中解析 model / pred_len / d_model。

    setting = task_vlm_modelid_model_data_ftX_slS_llL_plP_dmD_fsF_itr
    其中 modelid 含下划线，故按 'ft'/'sl'/'ll'/'pl'/'dm' 关键字定位。
    """
    info = {"setting": setting}
    m = re.search(r"_sl(\d+)_ll(\d+)_pl(\d+)_dm(\d+)", setting)
    if m:
        info["seq_len"], info["label_len"], info["pred_len"], info["d_model"] = (
            int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)))
    # model 名：取 setting 中 '_ft' 之前、最后一个下划线之后的字段；简化：直接在原始串里找
    # model 位于 task_vlm_modelid_MODEL_data 中，即从右往左找 '_data' 前的内容太绕，这里用已知白名单
    for name in ["TimeVLM", "PatchTST", "TimeLLM", "iTransformer", "TimeXer"]:
        if name in setting:
            info["model"] = name
            break
    if "wovlm" in setting:
        info["ablation"] = "w/o VLM"
    return info


def build_tables(rows, reference=None):
    md_lines = ["| Model | Horizon | d_model | MSE | MAE |", "|---|---|---|---|---|"]
    latex_lines = [
        r"\begin{tabular}{lccrr}",
        r"\toprule",
        r"Model & Horizon & $d_\text{model}$ & MSE & MAE \\",
        r"\midrule",
    ]
    for r in sorted(rows, key=lambda x: (x["setting"])):
        info = parse_setting(r["setting"])
        model = info.get("model", "?")
        if info.get("ablation"):
            model += " (w/o VLM)"
        pl = info.get("pred_len", "?")
        dm = info.get("d_model", "?")
        md_lines.append(f"| {model} | {pl} | {dm} | {r['mse']:.4f} | {r['mae']:.4f} |")
        latex_lines.append(
            f"{model} & {pl} & {dm} & {r['mse']:.4f} & {r['mae']:.4f} \\\\")
    latex_lines += [r"\bottomrule", r"\end{tabular}"]

    out_md = os.path.join("repro", "results_table.md")
    out_tex = os.path.join("repro", "results_table.tex")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write("\n".join(latex_lines) + "\n")
    print("=== Markdown ===")
    print("\n".join(md_lines))
    print("\n=== LaTeX ===")
    print("\n".join(latex_lines))
    print(f"\n[saved] {out_md} , {out_tex}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", type=str, default=None,
                    help="论文参考值 JSON: [{model, pred_len, mse, mae}, ...]")
    args = ap.parse_args()
    rows = parse_results()
    if not rows:
        print("没有可汇总的结果。请先运行训练脚本。")
        return
    build_tables(rows, args.reference)


if __name__ == "__main__":
    main()
