#!/usr/bin/env bash
# 并行分段下载 torch/torchvision cu118 wheel（阿里云镜像），加速国内下载。
set -e
BASE="https://mirrors.aliyun.com/pytorch-wheels/cu118"
OUTDIR=/c/Users/wjx/AppData/Local/Temp/tvwheels
mkdir -p "$OUTDIR"
cd "$OUTDIR"

TORCH_SIZE=2704239630   # torch-2.2.2+cu118-cp310-cp310-win_amd64.whl
CHUNKS=8

rm -f torch-2.2.2+cu118-cp310-cp310-win_amd64.whl chunk_*.part

CHUNK=$(( TORCH_SIZE / CHUNKS + 1 ))
for i in $(seq 0 $((CHUNKS - 1))); do
  START=$(( i * CHUNK ))
  END=$(( (i + 1) * CHUNK - 1 ))
  [ $END -ge $TORCH_SIZE ] && END=$(( TORCH_SIZE - 1 ))
  curl -sL --retry 3 --retry-delay 2 -r "${START}-${END}" \
    -o "chunk_${i}.part" \
    "${BASE}/torch-2.2.2%2Bcu118-cp310-cp310-win_amd64.whl" &
done
wait

: > torch-2.2.2+cu118-cp310-cp310-win_amd64.whl
for i in $(seq 0 $((CHUNKS - 1))); do
  cat "chunk_${i}.part" >> torch-2.2.2+cu118-cp310-cp310-win_amd64.whl
done
rm -f chunk_*.part

curl -sL --retry 3 --retry-delay 2 \
  -o "torchvision-0.17.2+cu118-cp310-cp310-win_amd64.whl" \
  "${BASE}/torchvision-0.17.2%2Bcu118-cp310-cp310-win_amd64.whl"

echo "=== downloaded ==="
ls -la *.whl
echo "expected torch bytes: $TORCH_SIZE"
