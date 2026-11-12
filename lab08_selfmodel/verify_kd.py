# -*- coding: utf-8 -*-
"""KD 实现的数值自检: KL 方向、T^2 缩放的梯度量级、alpha=0 与从零训练等价"""
import math
import torch
import torch.nn.functional as F

torch.manual_seed(0)
V, N = 322, 64          # 词表, 位置数
t_logits = torch.randn(N, V) * 2.0
s_logits = torch.randn(N, V) * 2.0


def kd_term(s_logits, t_logits, T, scale=True):
    s_lp = F.log_softmax(s_logits / T, dim=-1)
    t_p = F.softmax(t_logits / T, dim=-1)
    kd = F.kl_div(s_lp, t_p, reduction="batchmean")
    return kd * (T * T) if scale else kd


print("=== 1. KL 方向: 学生 == 教师 时 KL 应为 0 ===")
for T in (1.0, 2.0, 4.0, 8.0, 16.0):
    print("T=%4.1f  KL(教师||学生)=%.6f" % (T, kd_term(t_logits.clone(), t_logits, T)))

print("\n=== 2. 反方向 (KL(学生||教师)) 的值, 与上式不同, 说明方向被固定 ===")
s2 = t_logits + torch.randn_like(t_logits) * 0.5
a = kd_term(s2, t_logits, 4.0)
b = F.kl_div(F.log_softmax(t_logits / 4.0, dim=-1),
             F.softmax(s2 / 4.0, dim=-1), reduction="batchmean") * 16
print("KL(教师||学生)=%.4f ; KL(学生||教师)=%.4f" % (a, b))

print("\n=== 3. T^2 缩放后, KL 项对 logits 的梯度范数应基本不随 T 变化 ===")
print("%6s %14s %14s" % ("T", "∇(T^2·KL)", "∇(KL, 不乘T^2)"))
rows = []
for T in (1.0, 2.0, 4.0, 8.0, 16.0):
    gs = []
    for scale in (True, False):
        s = s2.clone().requires_grad_(True)
        kd_term(s, t_logits, T, scale).backward()
        gs.append(s.grad.norm().item())
    rows.append((T, gs[0], gs[1]))
    print("%6.1f %14.4f %14.4f" % (T, gs[0], gs[1]))

print("\n=== 4. 若漏乘 T^2, 梯度 ~ 1/T^2, 换温度需把 lr 放大 T^2 倍 ===")
base = rows[0][2]
for T, g_scaled, g_raw in rows:
    print("T=%4.1f : 漏乘时梯度仅为 T=1 的 %.4f 倍 (理论 1/T^2=%.4f)"
          % (T, g_raw / base, 1 / T**2))

print("\n=== 5. 数值单位: 温度越高, KL 项本身越小 (需 T^2 补偿) ===")
for T in (1.0, 2.0, 4.0, 8.0, 16.0):
    print("T=%4.1f  KL=%.5f  T^2·KL=%.5f" % (T, kd_term(s2, t_logits, T, False),
                                           kd_term(s2, t_logits, T, True)))
