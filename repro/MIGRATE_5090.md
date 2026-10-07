# RTX5090 迁移与训练手册

> 目标：把本地（RTX3050 4GB）已准备就绪的 Time-VLM 复现整套东西，迁移到 RTX5090（32GB，Blackwell sm_120）执行训练。
> 前置：已完成本机全部前置准备（见 `预执行报告.md`），本地**未执行任何训练**。

---

## 0. 迁移清单（拷到 5090 机器）

最小可运行集合：

| 文件/目录 | 说明 |
|---|---|
| 整个仓库（或 `git clone` 后 checkout `796e6ec`） | 代码 + `repro/` 脚本 + `scripts/` |
| `dataset/ETTh1.csv` | 2.6MB，MD5 `8381763947c85f4be6ac456c508460d6` |
| `dataset/prompt_bank/` | 7 个 prompt 文本（仓库自带） |
| `repro/` | 配置 + 运行/绘图/出表脚本 |
| （可选）本地 HF 缓存 `C:/Users/wjx/.cache/huggingface/hub/models--openai--clip-vit-base-patch32` | 581MB，拷到 5090 的 `~/.cache/huggingface/hub/` 可免重下 CLIP |

> 若用 git：`git clone https://github.com/CityMind-Lab/ICML25-TimeVLM && cd ICML25-TimeVLM && git checkout 796e6ec963788657207ea2b5553740993ea3ea2b`，再补拷 `dataset/ETTh1.csv` 和 `repro/`。

---

## 1. 5090 环境搭建（Linux）

> **目标机器：Ubuntu 24.04.4 LTS**。所有脚本（`run_5090.sh` 为 bash、Python 脚本均跨平台）可直接运行，无需改动。

**第一步：先确认 NVIDIA 驱动满足 Blackwell（RTX5090）要求**

```bash
nvidia-smi
```

- RTX 5090（Blackwell, sm_120）**必须** NVIDIA 驱动 ≥ **570.86.16**（CUDA 12.8）。Ubuntu 24.04 默认源的 `nvidia-driver-550` 只到 CUDA 12.4，**不支持 5090**。
- 若驱动过旧：`sudo apt install nvidia-driver-570`（或 575/580），或按老师的既有驱动环境处理。
- 确认 `nvidia-smi` 右上角显示 `CUDA Version: 12.8` 或更高（或至少 12.4 且已装对应 runtime），否则 torch 2.7.1+cu128 会报 `CUDA driver version is insufficient`。

**第二步：装 conda + 环境**

```bash
# 若无 conda：装 Miniconda（Linux x86_64）
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && bash Miniconda3-latest-Linux-x86_64.sh
conda create -y -n timevlm python=3.10
conda activate timevlm

# 1) 先装 torch（Blackwell sm_120 必须 cu128，torch>=2.7）
pip install "torch==2.7.1" "torchvision==0.22.1" --index-url https://download.pytorch.org/whl/cu128

# 2) 装其余依赖（torch 装好后再装，防止 reformer-pytorch 误升级 torch）
pip install -r <(grep -vE '^torch|^torchvision' requirements.txt) -i https://pypi.tuna.tsinghua.edu.cn/simple
# 补装 requirements.txt 缺失的包（本报告发现）
pip install pytorch-wavelets==1.3.0 "setuptools<81"

# 3) 验证
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast; from models.PatchTST import Model; from src.TimeVLM.model import Model; print('import OK')"
```

**版本注意（5090 与本地 3050 的差异）：**

| 包 | 本地 3050 | 5090 | 原因 |
|---|---|---|---|
| torch | 2.2.2+cu118 | **2.7.1+cu128** | Blackwell sm_120 无 2.2.2 kernel |
| torchvision | 0.17.2+cu118 | 0.22.1 | 配对 torch 2.7 |
| numpy | 1.23.5 | **≥1.24（建议 1.26.x）** | torch 2.7 要求 numpy≥1.24 |
| 其余 | 不变 | 不变 | transformers 4.46.3 / timm 1.0.15 等 |

**已知风险与预案（5090 上可能遇到）：**

1. **`torch.cuda.amp` 弃用告警**：仓库 `exp/exp_long_term_forecasting.py` 用 `torch.cuda.amp.autocast`/`GradScaler`，torch 2.7 里已弃用（仍可用，仅告警）。若报错，把 `torch.cuda.amp` 改成 `torch.amp`（`autocast` 需加 `device_type='cuda'`）。
2. **pytorch-wavelets 与 torch 2.7 兼容**：`pytorch_wavelets==1.3.0`（2020 老包）若在 torch 2.7 报 API 错误——**可安全注释掉** `layers/TimeSeries_To_Image.py` 和 `layers/Learnable_TimeSeries_To_Image.py` 里的 `from pytorch_wavelets import DWTForward` 行。因为默认 `learnable_image=True` 走的是 `LearnableTimeSeriesToImage`（卷积），**根本不调用 DWTForward**（它只用于未被接线的 FFT/小波变体）。
3. **reformer-pytorch 误升级 torch**：`reformer-pytorch==1.4.4` 有未固定 torch 依赖。务必**先装 torch 再装它**；若被升级，`pip install --no-deps "torch==2.7.1"` 回退。

---

## 2. CLIP 权重（二选一）

- **方式 A（免下载）**：把本地 HF 缓存目录 `models--openai--clip-vit-base-patch32` 整个拷到 5090 的 `~/.cache/huggingface/hub/`。
- **方式 B（在线下载）**：5090 若直连 huggingface.co 正常，直接跑训练会自动下载；若在国内网络，`export HF_ENDPOINT=https://hf-mirror.com`。

---

## 3. 运行训练

```bash
cd ICML25-TimeVLM
# 单卡：默认 GPU=0；多卡/指定卡：GPU=3 bash repro/run_5090.sh
bash repro/run_5090.sh
```

- 脚本按顺序跑 5 组：TimeVLM h96 → TimeVLM h720 → PatchTST h96 → PatchTST h720 → **w/o-VLM h720**（自动打补丁 → 训练 → 还原）。
- 想跳过消融：`bash repro/run_5090.sh --skip-ablation`。
- `num_workers=32` 若 5090 机器 CPU 核数少，改成 `num_workers=8`（脚本里全局替换）。
- 日志会打印 `Epoch: X | Train Loss ... Vali Loss ...`；早停 patience=5 自动生效。

---

## 4. 结果收集与汇报

训练完成后，产物：
- `result_long_term_forecast.txt`（各 setting 的 mse/mae）
- `./results/{setting}/`（`metrics.npy`/`pred.npy`/`true.npy`）
- `./test_results/{setting}/`（可视化 PDF）

一键出图/出表：

```bash
# 先改 repro/plot_results.py 里的 DEFAULT_SPEC，填入实际生成的 setting 串
python repro/plot_results.py    # 真值/TimeVLM/PatchTST 叠加曲线 + MSE/MAE 柱状图
python repro/make_table.py      # markdown + LaTeX 表（读 result_long_term_forecast.txt）
```

---

## 5. 验收判定（跑完对照）

把 `make_table.py` 输出的 MSE/MAE 与论文值对比（见 `paper_reference.json`）：

| Horizon | Time-VLM 论文 MSE/MAE | 复现成功阈值（±5%） |
|---|---|---|
| 96 | 0.361 / 0.386 | MSE 0.343–0.379 |
| 720 | 0.441 / 0.458 | MSE 0.419–0.463 |

- 通过：MSE 相对误差 ≤5%，且 h96 < h720，且 TimeVLM 优于仓库 PatchTST 基线，且 w/o-VLM 消融不优于 full。
- 异常终止：首 epoch Vali Loss > 2.0（归一化空间）即判发散，杀掉进程排查。

---

## 6. 常见问题

- **`cuda:1` 越界**：仓库 `--gpu` 默认 1，单卡必须 `--gpu 0`（`run_5090.sh` 已用 `GPU="${GPU:-0}"` 处理）。
- **`run.py --help` 崩溃**：`run.py:60` help 含 `%` 的 argparse bug，不影响训练，忽略即可。
- **找不到 ETTh1.csv**：确认 `./dataset/ETTh1.csv` 存在（root_path 是 `./dataset/`，不是 run.py 默认的 `./data/ETT/`）。
