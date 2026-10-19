#!/usr/bin/env python3
"""
Time-VLM 完整复现实验矩阵驱动器（2026-10-12）

覆盖：4 模型（TimeVLM/PatchTST/iTransformer/Time-LLM）× 5 数据集
（ETTh1/ETTh2/ETTm1/ETTm2/Exchange）× 4 horizons × full-shot/few-shot(10%)
+ zero-shot 迁移（TimeVLM/Time-LLM × 4 对 × 4 horizons）
+ 队尾 ECL 精简组（论文可比对的第 5 数据集，仅 full h96/h720）。

特性：断点续跑（按 result_long_term_forecast.txt 判重）、逐任务日志、
首 epoch 发散检测（Vali>3.0 或 NaN/Inf 即杀）、失败隔离、manifest 记录。

用法： /home/lv/miniconda3/envs/timevlm/bin/python -u repro/run_matrix.py
"""
import json
import os
import re
import signal
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
PY = sys.executable
RESULT = os.path.join(REPO, "result_long_term_forecast.txt")
LOGDIR = os.path.join(REPO, "logs", "matrix")
MANIFEST = os.path.join(REPO, "repro", "matrix_manifest.json")
DIVERGE = 8.0  # 归一化空间首 epoch Vali 阈值（NaN/Inf 单独判；3.0 对 few/zero-shot 误杀过多）

COMMON = dict(is_training=1,
              root_path="./dataset/", features="M", seq_len=512, label_len=48,
              des="Exp", itr=1, gpu=0, use_amp=True, train_epochs=15,
              batch_size=32, learning_rate=0.001, num_workers=32, patience=5)

# ---------------- 数据集定义 ----------------
DSETS = {
    "ETTh1": dict(data="ETTh1", path="ETTh1.csv", n_vars=7, freq="h"),
    "ETTh2": dict(data="ETTh2", path="ETTh2.csv", n_vars=7, freq="h"),
    "ETTm1": dict(data="ETTm1", path="ETTm1.csv", n_vars=7, freq="t"),
    "ETTm2": dict(data="ETTm2", path="ETTm2.csv", n_vars=7, freq="t"),
    "Exchange": dict(data="custom", path="exchange_rate.csv", n_vars=8, freq="d"),
    "ECL": dict(data="custom", path="ECL.csv", n_vars=321, freq="h"),
}

# TimeVLM 逐数据集逐 horizon 配置: (d_model, mem_gate, periodicity, dropout)
TVLM_FULL = {
    "ETTh1": {96: (32, False, 24, 0.1), 192: (32, False, 24, 0.1), 336: (64, True, 24, 0.1), 720: (256, True, 24, 0.3)},
    "ETTh2": {96: (64, False, 24, 0.2), 192: (64, False, 24, 0.3), 336: (128, False, 24, 0.3), 720: (32, False, 24, 0.3)},
    "ETTm1": {96: (64, True, 96, 0.2), 192: (32, True, 96, 0.2), 336: (128, True, 96, 0.3), 720: (32, True, 96, 0.2)},
    "ETTm2": {96: (32, True, 96, 0.2), 192: (32, True, 96, 0.2), 336: (32, True, 96, 0.2), 720: (32, True, 96, 0.2)},
    "Exchange": {96: (32, False, 7, 0.1), 192: (32, False, 7, 0.1), 336: (64, True, 7, 0.1), 720: (64, True, 7, 0.2)},
    "ECL": {96: (128, True, 24, 0.1), 192: (128, True, 24, 0.1), 336: (256, True, 24, 0.1), 720: (64, True, 24, 0.1)},
}
TVLM_FEW = {
    "ETTh1": {96: (64, True, 24, 0.1), 192: (64, True, 24, 0.1), 336: (64, True, 24, 0.1), 720: (64, True, 24, 0.1)},
    "ETTh2": {96: (64, True, 24, 0.1), 192: (128, True, 24, 0.1), 336: (256, True, 24, 0.1), 720: (512, True, 24, 0.1)},
    "ETTm1": {96: (64, True, 96, 0.1), 192: (128, True, 96, 0.1), 336: (256, True, 96, 0.1), 720: (256, True, 96, 0.1)},
    "ETTm2": {96: (64, True, 96, 0.1), 192: (128, True, 96, 0.1), 336: (256, True, 96, 0.1), 720: (256, True, 96, 0.1)},
    "Exchange": {96: (64, True, 7, 0.1), 192: (64, True, 7, 0.1), 336: (64, True, 7, 0.1), 720: (64, True, 7, 0.1)},
    "ECL": {96: (32, False, 24, 0.1), 192: (32, False, 24, 0.1), 336: (64, False, 24, 0.1), 720: (64, False, 24, 0.1)},
}
HORIZONS = [96, 192, 336, 720]


def tvlm_args(dset, h, few=False):
    d = DSETS[dset]
    cfg = (TVLM_FEW if few else TVLM_FULL)[dset][h]
    dm, mg, per, do = cfg
    return dict(**COMMON,
                model_id=f"{dset}_512_{h}" + ("_few" if few else ""),
                model="TimeVLM", data=d["data"], data_path=d["path"],
                pred_len=h, d_model=dm, enc_in=d["n_vars"], dec_in=d["n_vars"],
                c_out=d["n_vars"], image_size=56, norm_const=0.4, periodicity=per,
                three_channel_image=True, finetune_vlm=False, vlm_type="clip",
                use_mem_gate=mg, dropout=do, percent=0.1 if few else 1.0,
                freq=d["freq"], patch_len=16, stride=8, padding=8, top_k=5,
                patch_memory_size=100,
                task_name="few_shot_forecast" if few else "long_term_forecast")


def patchtst_args(dset, h, few=False):
    d = DSETS[dset]
    return dict(**COMMON, model_id=f"{dset}_512_{h}" + ("_few" if few else ""),
                model="PatchTST", data=d["data"], data_path=d["path"],
                pred_len=h, d_model=128, n_heads=8, e_layers=2, d_ff=768,
                dropout=0.1, activation="gelu", factor=3, enc_in=d["n_vars"],
                dec_in=d["n_vars"], c_out=d["n_vars"], patch_len=16, stride=8,
                percent=0.1 if few else 1.0, freq=d["freq"],
                task_name="few_shot_forecast" if few else "long_term_forecast")


def itrans_args(dset, h, few=False):
    d = DSETS[dset]
    return dict(**COMMON, model_id=f"{dset}_512_{h}" + ("_few" if few else ""),
                model="iTransformer", data=d["data"], data_path=d["path"],
                pred_len=h, d_model=128, n_heads=8, e_layers=2, d_ff=128,
                dropout=0.1, activation="gelu", factor=1, enc_in=d["n_vars"],
                dec_in=d["n_vars"], c_out=d["n_vars"],
                percent=0.1 if few else 1.0, freq=d["freq"],
                task_name="few_shot_forecast" if few else "long_term_forecast")


def timellm_args(dset, h, few=False):
    d = DSETS[dset]
    a = dict(**COMMON, model_id=f"{dset}_512_{h}" + ("_few" if few else ""),
             model="TimeLLM", data=d["data"], data_path=d["path"],
             pred_len=h, d_model=32, d_ff=128, n_heads=8, e_layers=2,
             dropout=0.1, activation="gelu", enc_in=d["n_vars"],
             dec_in=d["n_vars"], c_out=d["n_vars"], patch_len=16, stride=8,
             llm_model="GPT2", llm_layers=1, prompt_domain=1,
             percent=0.1 if few else 1.0, freq=d["freq"],
             task_name="few_shot_forecast" if few else "long_term_forecast")
    a.pop("use_amp")  # 冒烟实测：Time-LLM + AMP 在 epoch 2 产生 NaN，禁用
    return a


def zero_args(model, src, tgt, h):
    d = DSETS[src]
    base = dict(**COMMON, task_name="zero_shot_forecast",
                model_id=f"{src}_{tgt}_512_{h}", model=model,
                data=d["data"], data_path=d["path"], pred_len=h, d_model=64,
                enc_in=d["n_vars"], dec_in=d["n_vars"], c_out=d["n_vars"],
                target_data=tgt, target_root_path="./dataset/",
                target_data_path=DSETS[tgt]["path"], freq=d["freq"],
                percent=1.0, patch_len=16, stride=8, padding=8, top_k=5,
                patch_memory_size=100)
    if model == "TimeVLM":
        base.update(image_size=56, norm_const=0.4, periodicity=24,
                    three_channel_image=True, finetune_vlm=False,
                    vlm_type="clip", use_mem_gate=True, dropout=0.1)
    if model == "TimeLLM":
        base.pop("use_amp")  # Time-LLM 禁用 AMP（冒烟实测 NaN）
    return base


def build_tasks():
    tasks = []
    models = [tvlm_args, patchtst_args, itrans_args, timellm_args]
    order = ["Exchange", "ETTh1", "ETTh2", "ETTm1", "ETTm2"]
    for dset in order:
        for few in [False, True]:
            for mf in models:
                mname = mf.__name__.replace("_args", "")
                for h in HORIZONS:
                    a = mf(dset, h, few)
                    tasks.append(dict(name=f"{'few' if few else 'full'}_{mname}_{a['model_id']}",
                                      model=mname,
                                      dataset=dset, horizon=h,
                                      setting="few" if few else "full",
                                      args=a, timeout=6 * 3600))
    for model, src, tgt in [("TimeVLM", "ETTh1", "ETTh2"), ("TimeVLM", "ETTh2", "ETTh1"),
                            ("TimeVLM", "ETTm1", "ETTm2"), ("TimeVLM", "ETTm2", "ETTm1"),
                            ("TimeLLM", "ETTh1", "ETTh2"), ("TimeLLM", "ETTh2", "ETTh1"),
                            ("TimeLLM", "ETTm1", "ETTm2"), ("TimeLLM", "ETTm2", "ETTm1")]:
        for h in HORIZONS:
            a = zero_args(model, src, tgt, h)
            tasks.append(dict(name=f"zero_{model}_{a['model_id']}", model=model,
                              dataset=f"{src}->{tgt}", horizon=h,
                              setting="zero", args=a, timeout=6 * 3600))
    # ECL 精简组（队尾）：TimeVLM + 3 基线，仅 full h96/h720
    for mf in [tvlm_args, patchtst_args, itrans_args, timellm_args]:
        mname = mf.__name__.replace("_args", "")
        for h in [96, 720]:
            a = mf("ECL", h, few=False)
            if mname == "timellm":
                # 321 通道：TSL 官方 Time-LLM ECL 惯例 seq96；B*N 展平下 seq512+batch32 OOM
                a["seq_len"] = 96
                a["batch_size"] = 16
                a["model_id"] = f"ECL_96_{h}"
            tasks.append(dict(name=f"ecl_full_{mname}_{a['model_id']}", model=mname,
                              dataset="ECL", horizon=h, setting="full",
                              args=a, timeout=10 * 3600))
    return tasks


def load_done():
    if not os.path.exists(RESULT):
        return set()
    with open(RESULT) as f:
        text = f.read()
    return set(re.findall(r"^\S+", text, re.M))


STORE_TRUE_ARGS = {"use_amp"}  # argparse action='store_true'，不能带值


def cmd_of(a):
    parts = [PY, "-u", "run.py"]
    for k, v in a.items():
        if k in STORE_TRUE_ARGS:
            if v:
                parts.append(f"--{k}")
            continue
        if isinstance(v, bool):
            parts += [f"--{k}", "True" if v else "False"]
        else:
            parts += [f"--{k}", str(v)]
    return parts


def run_task(t, logfile):
    t0 = time.time()
    os.makedirs(LOGDIR, exist_ok=True)
    with open(logfile, "w") as lf:
        proc = subprocess.Popen(cmd_of(t["args"]), stdout=lf, stderr=subprocess.STDOUT)
        first_val = None
        killed = False
        while proc.poll() is None:
            time.sleep(15)
            try:
                with open(logfile, errors="ignore") as f:
                    text = f.read()
            except OSError:
                continue
            if first_val is None:
                m = re.search(r"Vali Loss:\s*([0-9.eE+-]+|nan|inf)", text)
                if m:
                    v = m.group(1)
                    first_val = float(v) if v not in ("nan", "inf") else float("inf")
                    if first_val > DIVERGE:
                        proc.send_signal(signal.SIGKILL)
                        killed = True
            # 只检查 epoch 汇总行的 loss（早停的 "(inf --> X)" 消息是正常的）
            if re.search(r"(Train|Vali|Test) Loss:\s*(nan|inf)", text, re.I):
                proc.send_signal(signal.SIGKILL)
                killed = True
        rc = proc.returncode
        with open(logfile, errors="ignore") as f:
            text = f.read()
    done = "mse:" in text or "MSE:" in text  # zero-shot exp 用大写 MSE:
    status = "DIVERGED" if killed else ("DONE" if done and rc == 0 else "FAILED")
    return status, first_val, time.time() - t0


def main():
    tasks = build_tasks()
    done = load_done()
    manifest = {}
    if os.path.exists(MANIFEST):
        with open(MANIFEST) as f:
            manifest = json.load(f)
    print(f"[matrix] {len(tasks)} tasks, {len(done)} settings already in result file")
    for i, t in enumerate(tasks):
        tag = f"[{i + 1}/{len(tasks)}] {t['name']}"
        if t["name"] in manifest and manifest[t["name"]]["status"] == "DONE":
            print(f"{tag} SKIP (manifest done)")
            continue
        logfile = os.path.join(LOGDIR, t["name"] + ".log")
        print(f"{tag} RUN {t['model']} {t['dataset']} h{t['horizon']} {t['setting']} -> {logfile}", flush=True)
        status, first_val, secs = run_task(t, logfile)
        manifest[t["name"]] = dict(status=status, first_val=first_val,
                                   secs=round(secs), time=time.strftime("%m-%d %H:%M"))
        with open(MANIFEST, "w") as f:
            json.dump(manifest, f, indent=1, ensure_ascii=False)
        print(f"{tag} {status} (first_val={first_val}, {secs / 60:.1f} min)", flush=True)
        if status == "FAILED":
            time.sleep(30)  # 失败后留缓冲，避免连续错误刷屏
    n = manifest
    print("[matrix] finished. summary:",
          {s: sum(1 for v in n.values() if v["status"] == s) for s in ("DONE", "FAILED", "DIVERGED")})


if __name__ == "__main__":
    main()
