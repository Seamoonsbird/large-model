"""生成报告图表：架构图、训练曲线、混淆矩阵、压缩 trade-off、预测样例。"""
import json, os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from data_utils import get_loaders
from models import SmallCNN

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.linewidth': 0.8,
                     'figure.dpi': 200, 'savefig.bbox': 'tight'})
OUT = '../figures'
os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, GREY = '#2f6fed', '#e8833a', '#8a8f98'


def box(ax, x, y, w, h, text, fc='#eef3fd', ec=BLUE, fs=8.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.08',
                                fc=fc, ec=ec, lw=1.0))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs,
            fontweight='bold' if bold else 'normal', color='#1a1a1a')


def arrow(ax, p1, p2, color=GREY):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle='-|>', mutation_scale=9, lw=0.9, color=color))


def fig_arch():
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3.4); ax.axis('off')
    ys, h = 1.95, 0.62
    box(ax, 0.1, ys, 1.0, h, 'Input\n28×28×1', fc='#f2f2f2', ec=GREY)
    box(ax, 1.35, ys, 1.35, h, 'Conv 3×3\n32 ch + BN')
    box(ax, 2.95, ys, 1.35, h, 'Conv 3×3\n64 ch + BN')
    box(ax, 4.55, ys, 0.75, h, 'Pool\n2×2', fc='#fff4e8', ec=ORANGE)
    box(ax, 5.55, ys, 1.35, h, 'Conv 3×3\n128 ch + BN')
    box(ax, 7.15, ys, 0.75, h, 'Pool\n2×2', fc='#fff4e8', ec=ORANGE)
    box(ax, 8.15, ys, 0.72, h, 'GAP\n128-d', fc='#fff4e8', ec=ORANGE)
    box(ax, 9.05, ys, 0.82, h, 'FC\n10-way', fc='#eaf7ee', ec='#3f9c5a')
    for a, b in [(1.10, 1.35), (2.70, 2.95), (4.30, 4.55), (5.30, 5.55), (6.90, 7.15), (7.90, 8.15), (8.87, 9.05)]:
        arrow(ax, (a, ys + h / 2), (b, ys + h / 2))
    ax.text(5.0, 3.05, 'Visual backbone (CNN, Lesson 3)', ha='center', fontsize=9, fontweight='bold')
    ax.text(0.1, 1.35, 'Compression pipeline (Lesson 15)', fontsize=9, fontweight='bold')
    steps = [('Float32\nteacher', 0.1), ('Knowledge\ndistillation', 2.1), ('Structured\npruning', 4.05), ('Static INT8\nPTQ', 6.0), ('Edge\nmodel', 7.95)]
    for t, x in steps:
        box(ax, x, 0.55, 1.55, 0.6, t, fc='#fdf1e7', ec=ORANGE, fs=8)
    for x in [1.65, 3.65, 5.60, 7.55]:
        arrow(ax, (x, 0.85), (x + 0.45, 0.85), color=ORANGE)
    ax.annotate('', xy=(9.6, 1.88), xytext=(9.6, 1.2), arrowprops=dict(arrowstyle='-|>', color=GREY, lw=0.9))
    ax.text(9.72, 1.54, 'deploy', fontsize=7.5, rotation=90, va='center', color=GREY)
    fig.savefig(os.path.join(OUT, 'fig1_architecture.png')); plt.close(fig)


def fig_curves():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    files = [('cnn_mnist', 'CNN / MNIST', BLUE), ('vit_mnist', 'TinyViT / MNIST', ORANGE),
             ('cnn_fashion', 'CNN / Fashion-MNIST', '#3f9c5a')]
    for tag, label, c in files:
        p = f'../results/{tag}_history.json'
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        ep = [h['epoch'] for h in d['history']]
        axes[0].plot(ep, [h['train_loss'] for h in d['history']], '-o', ms=3, color=c, label=label)
        axes[1].plot(ep, [h['val_acc'] * 100 for h in d['history']], '-o', ms=3, color=c, label=label)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Training loss'); axes[0].set_title('(a) Optimization', fontsize=9)
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Validation accuracy (%)'); axes[1].set_title('(b) Generalization', fontsize=9)
    axes[1].set_ylim(50, 100)
    for a in axes:
        a.grid(alpha=0.25, lw=0.6); a.legend(fontsize=7, frameon=False)
    fig.savefig(os.path.join(OUT, 'fig2_training_curves.png')); plt.close(fig)


def fig_confusion():
    d = torch.load('../results/final_preds.pt', weights_only=False)
    cm = d['cm'].numpy()
    fig, ax = plt.subplots(figsize=(3.9, 3.4))
    im = ax.imshow(cm, cmap='Blues')
    for i in range(10):
        for j in range(10):
            if cm[i, j]:
                ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=5.6,
                        color='white' if cm[i, j] > cm.max() / 2 else '#1a1a1a')
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title('Final INT8 model (MNIST)', fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    fig.savefig(os.path.join(OUT, 'fig3_confusion.png')); plt.close(fig)


def fig_tradeoff():
    m = json.load(open('../results/metrics.json'))
    order = [('cnn_fp32', 'CNN FP32'), ('prune50', 'Pruned 50%'), ('kd_student', 'KD student'),
             ('cnn_int8', 'CNN INT8'), ('kd_int8', 'KD+INT8'), ('ours', 'Ours (KD+Prune+INT8)')]
    labels = [l for k, l in order if k in m]
    accs = [m[k]['acc'] * 100 for k, _ in order if k in m]
    sizes = [m[k]['mb'] for k, _ in order if k in m]
    lats = [m[k]['lat'] for k, _ in order if k in m]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55), sharey=True)
    cols = ['#b9c2cf', '#8fa8d8', '#6f8fd0', '#4a72b8', '#2f5fa8', BLUE]
    axes[0].barh(labels, accs, color=cols)
    axes[0].set_xlim(0, 108); axes[0].set_xlabel('Accuracy (%)')
    for i, v in enumerate(accs):
        axes[0].text(v - 1.2, i, f'{v:.2f}', va='center', ha='right', fontsize=6.5, color='white')
    axes[1].barh(labels, sizes, color=cols); axes[1].set_xlim(0, max(sizes) * 1.35)
    axes[1].set_xlabel('Model size (MB)')
    for i, v in enumerate(sizes):
        axes[1].text(v + max(sizes) * 0.02, i, f'{v:.3f}', va='center', fontsize=6.5)
    axes[2].barh(labels, lats, color=cols); axes[2].set_xlim(0, max(lats) * 1.25)
    axes[2].set_xlabel('Latency (ms, batch=1)')
    for i, v in enumerate(lats):
        axes[2].text(v + max(lats) * 0.02, i, f'{v:.2f}', va='center', fontsize=6.5)
    for a in axes:
        a.tick_params(labelsize=6.5); a.grid(axis='x', alpha=0.25, lw=0.6)
    axes[0].set_title('(a) Accuracy', fontsize=8.5); axes[1].set_title('(b) Size', fontsize=8.5); axes[2].set_title('(c) CPU latency', fontsize=8.5)
    fig.subplots_adjust(left=0.22, wspace=0.08)
    fig.savefig(os.path.join(OUT, 'fig4_tradeoff.png')); plt.close(fig)


def fig_samples():
    _, te, _, te_ds = get_loaders('../data', '', batch_size=256)
    x_all, y_all = te_ds
    d = torch.load('../results/final_preds.pt', weights_only=False)
    pred_all, logits = d['pred'], d['logits']
    conf_all = torch.softmax(logits, 1).max(1).values
    idx = []
    for c in range(10):
        idx += (y_all == c).nonzero().flatten()[:3].tolist()
    xs = x_all[idx]; ys = y_all[idx]
    pred, conf = pred_all[idx], conf_all[idx]
    fig, axes = plt.subplots(3, 10, figsize=(7.2, 2.5))
    for ax, i in zip(axes.flat, range(len(idx))):
        ax.imshow(xs[i, 0] * 0.3081 + 0.1307, cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'{int(pred[i])} ({conf[i]*100:.0f}%)', fontsize=6,
                     color='#1a1a1a' if pred[i] == ys[i] else '#c0392b')
        ax.axis('off')
    fig.suptitle('Demo outputs: predicted digit (confidence), red = error', fontsize=8, y=1.02)
    fig.savefig(os.path.join(OUT, 'fig5_samples.png')); plt.close(fig)


if __name__ == '__main__':
    fig_arch()
    fig_curves()
    if os.path.exists('../results/final_preds.pt'):
        fig_confusion(); fig_samples()
    if os.path.exists('../results/metrics.json'):
        fig_tradeoff()
    print('figures done:', os.listdir(OUT))
