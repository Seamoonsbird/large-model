# -*- coding: utf-8 -*-
"""从 runs_distill/*.json 生成报告用图"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = ["Noto Serif CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(ROOT, "runs_distill")
FIG = os.path.join(R, "figures")
os.makedirs(FIG, exist_ok=True)


def load(tag):
    with open(os.path.join(R, "results_%s.json" % tag), encoding="utf-8") as f:
        return json.load(f)


allr = load("all")
if not allr["teacher"].get("curve"):
    tr = load("teacher")
    allr["teacher"]["curve"] = tr["teacher"]["curve"]
abr = load("ablate")
swr = load("sweep")

# ---------------- 图1: 训练损失曲线 ----------------
fig, ax = plt.subplots(figsize=(7.2, 3.8))
for k, lab, c in (("teacher", "教师 0.74M", "#333333"),
                  ("scratch", "学生·从零 0.15M", "#C0504D"),
                  ("kd", "学生·蒸馏 T=4, α=0.5", "#4F81BD")):
    cur = allr[k].get("curve_ce") or allr[k]["curve"]
    ls = "--" if k == "teacher" else "-"
    ax.plot(range(1, len(cur) + 1), cur, label=lab, color=c, ls=ls,
            linewidth=1.6 if k == "teacher" else 1.2,
            zorder=5 if k == "teacher" else 2)
ax.set_xlabel("训练步数 (iteration)")
ax.set_ylabel("训练交叉熵 CE")
ax.set_title("教师 / 从零学生 / 蒸馏学生的训练损失曲线")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.set_xlim(0, 205)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig1_training_curves.png"), dpi=170)
plt.close(fig)

# ---------------- 图2: T x alpha 消融 ----------------
rows = abr["ablation"]
Ts = sorted({r["T"] for r in rows})
As = sorted({r["alpha"] for r in rows})
M = np.zeros((len(Ts), len(As)))
for r in rows:
    M[Ts.index(r["T"]), As.index(r["alpha"])] = r["val_loss"]

fig, ax = plt.subplots(1, 2, figsize=(10, 4.0), gridspec_kw={"width_ratios": [1, 1.15]})
for T in Ts:
    xs = [r["alpha"] for r in rows if r["T"] == T]
    ys = [r["val_loss"] for r in rows if r["T"] == T]
    ax[0].plot(xs, ys, marker="o", label="T=%d" % T, linewidth=1.3)
ax[0].axhline(1.5360, ls="--", c="gray", lw=1, label="从零训练基线 1.5360")
ax[0].set_xlabel("蒸馏权重 α")
ax[0].set_ylabel("留出集损失")
ax[0].set_title("同温度下 α 的影响")
ax[0].legend(fontsize=8)
ax[0].grid(alpha=0.3)

im = ax[1].imshow(M, cmap="viridis_r", aspect="auto")
for i in range(len(Ts)):
    for j in range(len(As)):
        ax[1].text(j, i, "%.4f" % M[i, j], ha="center", va="center",
                   fontsize=9, color="w" if M[i, j] > M.mean() else "k")
ax[1].set_xticks(range(len(As)), ["α=%.1f" % a for a in As])
ax[1].set_yticks(range(len(Ts)), ["T=%d" % t for t in Ts])
ax[1].set_title("T × α 留出集损失热力图 (80 步)")
best = np.unravel_index(np.argmin(M), M.shape)
ax[1].add_patch(plt.Rectangle((best[1] - 0.5, best[0] - 0.5), 1, 1, fill=False,
                              edgecolor="red", lw=2))
fig.colorbar(im, ax=ax[1], shrink=0.85, label="val loss")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig2_ablation.png"), dpi=170)
plt.close(fig)

# ---------------- 图3: 留出集损失随训练步数的变化 ----------------
tab = swr["sweep"]["table"]
fig, ax = plt.subplots(figsize=(7.2, 3.8))
ax.plot([r["iter"] for r in tab], [r["scratch"] for r in tab], marker="o", ms=4,
        label="学生·从零训练", color="#C0504D")
ax.plot([r["iter"] for r in tab], [r["kd"] for r in tab], marker="s", ms=4,
        label="学生·蒸馏 T=4, α=0.5", color="#4F81BD")
ax.axhline(allr["teacher"]["val_loss"], ls="--", c="gray", lw=1)
ax.text(8, allr["teacher"]["val_loss"] + 0.02, "教师留出集损失 %.4f"
        % allr["teacher"]["val_loss"], fontsize=8, color="gray")
ax.axvline(120, ls=":", c="green", lw=1)
ax.text(122, 4.35, "默认学生步数 120", fontsize=8, color="green")
ax.set_xlabel("训练步数")
ax.set_ylabel("留出集损失")
ax.set_title("蒸馏与从零训练的留出集损失随步数变化")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.set_ylim(0.9, 5.0)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig3_val_vs_steps.png"), dpi=170)
plt.close(fig)

print("figures saved:", os.listdir(FIG))
