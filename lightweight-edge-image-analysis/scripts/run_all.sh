#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../code"
python3 train.py --model cnn --data mnist --epochs 5
python3 compress.py
python3 train.py --model vit --data mnist --epochs 4
python3 train.py --model cnn_small --data mnist --epochs 3
python3 train.py --model cnn --data fashion --epochs 5
python3 eval_extra.py
python3 figures.py
echo "done: see ../results/metrics.json and ../figures/"
