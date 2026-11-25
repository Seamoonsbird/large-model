#!/usr/bin/env bash
# 下载 MNIST 与 Fashion-MNIST 的原始 IDX 文件（约 41 MB），放到 ./data 下。
set -e
cd "$(dirname "$0")/../data"

MIRROR_M="https://storage.googleapis.com/cvdf-datasets/mnist"
MIRROR_F="https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion"

for f in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
  [ -s "$f.gz" ] || curl -fL --retry 3 -o "$f.gz" "$MIRROR_M/$f.gz"
  [ -s "fashion-$f.gz" ] || curl -fL --retry 3 -o "fashion-$f.gz" "$MIRROR_F/$f.gz"
done

ls -lh *.gz
