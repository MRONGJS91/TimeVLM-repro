"""
消融补丁：将 Time-VLM 仓库中已存在但未接入的 `w_out_visual` / `w_out_text` 死参数
接入 src/TimeVLM/model.py 的 forward()，用于 "Time-VLM w/o VLM" 消融实验。

原理：在 forward() 拿到 vision/text 嵌入后，若 config 置位，则将其置零，
使 multimodal 分支接收不到 VLM 信息，等价于去掉 VLM（视觉+文本）编码器。

用法：
    python repro/patch_w_out_vlm.py            # 打补丁（幂等，重复执行安全）
    python repro/patch_w_out_vlm.py --revert   # 还原
"""
import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(REPO_ROOT, "src", "TimeVLM", "model.py")

ANCHOR = "        vision_embeddings, text_embeddings = self.vlm_manager.process_inputs(B, images, prompts)"
MARKER = "w_out_visual"
INJECT = (
    "        vision_embeddings, text_embeddings = self.vlm_manager.process_inputs(B, images, prompts)\n"
    "        # ---- w/o VLM ablation (injected by repro/patch_w_out_vlm.py) ----\n"
    "        if getattr(self.config, 'w_out_visual', False):\n"
    "            vision_embeddings = torch.zeros_like(vision_embeddings)\n"
    "        if getattr(self.config, 'w_out_text', False):\n"
    "            text_embeddings = torch.zeros_like(text_embeddings)\n"
)


def patch():
    with open(MODEL_PATH, encoding="utf-8") as f:
        src = f.read()
    if "w_out_visual', False" in src and "torch.zeros_like(vision_embeddings)" in src:
        print("[patch] 已打补丁（幂等），无需重复执行。")
        return
    if ANCHOR not in src:
        print(f"[patch] 错误：未找到锚点行。请确认 model.py 未改动。")
        sys.exit(1)
    src = src.replace(ANCHOR, INJECT, 1)
    with open(MODEL_PATH, "w", encoding="utf-8") as f:
        f.write(src)
    print("[patch] 已注入 w/o-VLM 消融开关到", MODEL_PATH)


def revert():
    with open(MODEL_PATH, encoding="utf-8") as f:
        src = f.read()
    if INJECT not in src:
        print("[revert] 未检测到补丁，无需还原。")
        return
    src = src.replace(INJECT, ANCHOR, 1)
    with open(MODEL_PATH, "w", encoding="utf-8") as f:
        f.write(src)
    print("[revert] 已还原 model.py")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--revert", action="store_true")
    args = ap.parse_args()
    revert() if args.revert else patch()
