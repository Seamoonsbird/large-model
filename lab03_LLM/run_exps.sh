#!/bin/bash
cd /home/user/Doubao/chats/38444858498684162/hw3_min_llm
set -x
python3 min_llm.py --iters 1000 --lr 1e-2   --save_ckpt exp1a.pt > run_exp1a.log 2>&1
python3 min_llm.py --iters 1000 --lr 1e-4   --save_ckpt exp1b.pt > run_exp1b.log 2>&1
python3 min_llm.py --iters 1000 --n_layer 1               --save_ckpt exp2a.pt > run_exp2a.log 2>&1
python3 min_llm.py --iters 1000 --n_layer 4               --save_ckpt exp2b.pt > run_exp2b.log 2>&1
python3 min_llm.py --iters 1000 --n_embd 64  --n_head 2   --save_ckpt exp3a.pt > run_exp3a.log 2>&1
python3 min_llm.py --iters 1000 --n_embd 256 --n_head 8   --save_ckpt exp3b.pt > run_exp3b.log 2>&1
python3 min_llm.py --iters 1000 --block_size 32           --save_ckpt exp4.pt  > run_exp4.log 2>&1
python3 min_llm.py --iters 1000 --no_pos                   --save_ckpt exp5.pt  > run_exp5.log 2>&1
echo "ALL_EXP_DONE"
