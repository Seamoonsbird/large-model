#!/bin/bash
# 2 核：每次跑 2 组，每组 1 线程，打满两核
cd /home/user/Doubao/chats/38441736218918402/code
run () { python3 run_exp.py --group $1 --seed 42 --threads 1 > /tmp/log_$1.txt 2>&1; echo "DONE $1"; }
run baseline &
run exp5 &
wait
run exp1 &
run exp6 &
wait
run exp2 &
run exp3 &
wait
run exp4 &
wait
echo "ALL_GROUPS_DONE"
