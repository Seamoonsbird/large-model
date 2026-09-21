"""
ConvLSTM 弹跳小球时空预测实验（作业二）。
修正了指南参考代码中的 bug：
  - ConvLSTM.forward 中 return 的缩进（原参考代码 return 在时间循环内，只跑了一帧）；
  - if __name__ == "__name__" 拼写；
  - 数据划分重叠：原参考代码 test=data[1800:] 与 train=data[:2000] 重叠 200 条；
    这里 n_seq=2200，train=data[:2000]、test=data[2000:]，互不重叠。
用法：
  python run_exp.py --group baseline          # 跑指定组，写 metrics/<group>.json 与 figures/<fig>.png
  python run_exp.py --group best --trials 3   # 最优组合跑 3 次
所有运行固定随机种子（默认 42），保证可复现。
"""
import argparse
import json
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

import torch
import torch.nn as nn

# ---------- 中文字体（系统无 Microsoft YaHei，改用 Noto Sans CJK SC） ----------
_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(_FONT_PATH):
    font_manager.fontManager.addfont(_FONT_PATH)
    _zh = font_manager.FontProperties(fname=_FONT_PATH).get_name()
    plt.rcParams["font.sans-serif"] = [_zh, "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "..", "figures")
METRIC_DIR = os.path.join(HERE, "..", "metrics")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(METRIC_DIR, exist_ok=True)


# ---------------- 数据合成 ----------------
def make_sequences(n_seq=2200, T=10, size=32, r=2, seed=42):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    seqs = np.zeros((n_seq, T, size, size), np.float32)
    for s in range(n_seq):
        x, y = rng.uniform(4, size - 5, 2)
        vx, vy = rng.choice([-1, 1], 2) * rng.uniform(0.8, 1.6, 2)
        for t in range(T):
            x, y = x + vx, y + vy
            if x < r or x > size - r:
                vx = -vx
            if y < r or y > size - r:
                vy = -vy
            seqs[s, t] = ((xx - x) ** 2 + (yy - y) ** 2 <= r * r)
    return seqs[..., None]  # (N, T, 32, 32, 1)


def build_dataset(k_in=4, seed=42):
    """train[:2000] / test[2000:]，互不重叠。返回 (train_x, train_y, test_x, test_y)。"""
    data = make_sequences(n_seq=2200, T=10, size=32, r=2, seed=seed)
    # (N, T, H, W, 1) -> (N, T, 1, H, W)
    data = torch.tensor(data).permute(0, 1, 4, 2, 3)
    train_x = data[:2000, :k_in]
    train_y = data[:2000, k_in].squeeze(1)   # (N,32,32)
    test_x = data[2000:, :k_in]
    test_y = data[2000:, k_in].squeeze(1)
    return train_x, train_y, test_x, test_y


# ---------------- 模型 ----------------
class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hid_ch, k=3):
        super().__init__()
        self.conv = nn.Conv2d(in_ch + hid_ch, 4 * hid_ch, k, padding=k // 2)
        self.hid = hid_ch

    def forward(self, x, state):
        h, c = state
        z = self.conv(torch.cat([x, h], dim=1))
        i, f, g, o = z.chunk(4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        c_new = f * c + i * torch.tanh(g)
        h_new = o * torch.tanh(c_new)
        return h_new, c_new


class ConvLSTM(nn.Module):
    def __init__(self, in_ch=1, hid=32, k=3, layers=1):
        super().__init__()
        chs = [in_ch] + [hid] * layers
        self.cells = nn.ModuleList(
            [ConvLSTMCell(chs[i], chs[i + 1], k) for i in range(layers)]
        )
        self.out = nn.Conv2d(hid, 1, 3, padding=1)
        self.hid = hid

    def forward(self, x):
        # x: (B, T, C, H, W)
        b, tsteps, _, hsize, wsize = x.shape
        states = [
            (torch.zeros(b, c.hid, hsize, wsize), torch.zeros(b, c.hid, hsize, wsize))
            for c in self.cells
        ]
        for t in range(tsteps):
            xt = x[:, t]  # (B,C,H,W)
            for j, cell in enumerate(self.cells):
                states[j] = cell(xt, states[j])
                xt = states[j][0]
        return self.out(states[-1][0]).squeeze(1)  # (B,32,32)


class FlattenLSTM(nn.Module):
    """实验5结构对照：展平每帧为1024维，nn.LSTM(1024,256) -> Linear(256->1024) -> 32x32。"""
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1024, 256)
        self.fc = nn.Linear(256, 1024)

    def forward(self, x):
        b, tsteps, _, h, w = x.shape
        seq = x.reshape(b, tsteps, -1).permute(1, 0, 2)  # (T,B,1024)
        out, _ = self.lstm(seq)
        y = self.fc(out[-1])  # (B,1024)
        return y.reshape(b, 32, 32)


# ---------------- 训练 ----------------
def train_one(model, train_x, train_y, test_x, test_y, loss_name="mse",
              lr=0.001, epochs=5, bs=64, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    loss_fn = nn.MSELoss() if loss_name == "mse" else nn.L1Loss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = train_x.shape[0]
    hist = {"epoch": [], "train_loss": [], "test_mse": [], "test_mae": []}
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        ep_loss = 0.0
        nb = 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = loss_fn(model(train_x[idx]), train_y[idx])
            loss.backward()
            opt.step()
            ep_loss += loss.item()
            nb += 1
        model.eval()
        with torch.no_grad():
            pred = model(test_x)
            mse = (pred - test_y).pow(2).mean().item()
            mae = (pred - test_y).abs().mean().item()
        hist["epoch"].append(ep + 1)
        hist["train_loss"].append(ep_loss / nb)
        hist["test_mse"].append(mse)
        hist["test_mae"].append(mae)
    dt = time.time() - t0
    # 最终测试指标（最后一轮）
    final_mse = hist["test_mse"][-1]
    final_mae = hist["test_mae"][-1]
    params = sum(p.numel() for p in model.parameters())
    return hist, dt, final_mse, final_mae, params, model


def plot_curve(hist, title, out_png):
    fig, ax1 = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    ax1.plot(hist["epoch"], hist["train_loss"], "o-", color="#1f77b4", label="训练损失")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("训练损失", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax2 = ax1.twinx()
    ax2.plot(hist["epoch"], hist["test_mse"], "s--", color="#d62728", label="测试 MSE")
    ax2.set_ylabel("测试 MSE", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    plt.title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_pred(model, test_x, test_y, out_png, n_samples=3):
    model.eval()
    with torch.no_grad():
        pred = model(test_x[:n_samples])
    fig, axes = plt.subplots(n_samples, 3, figsize=(7.5, 7.5), dpi=150)
    for i in range(n_samples):
        inp = test_x[i]           # (T,1,32,32)
        truth = test_y[i]         # (32,32)
        p = pred[i]               # (32,32)
        mse_i = ((p - truth) ** 2).mean().item()
        # 输入帧：把前 k_in 帧叠成一张（取最后一帧代表，或叠加最大投影）
        inp_img = inp[-1, 0].numpy()
        axes[i, 0].imshow(inp_img, cmap="gray", vmin=0, vmax=1)
        axes[i, 0].set_title("输入末帧")
        axes[i, 1].imshow(truth.numpy(), cmap="gray", vmin=0, vmax=1)
        axes[i, 1].set_title("真实帧")
        axes[i, 2].imshow(p.numpy(), cmap="gray", vmin=0, vmax=1)
        axes[i, 2].set_title(f"预测帧\nMSE={mse_i:.5f}")
        for j in range(3):
            axes[i, j].axis("off")
    plt.suptitle("输入帧 | 真实帧 | 预测帧 对比", y=0.99)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# ---------------- 组配置 ----------------
GROUP_CONFIG = {
    "baseline": dict(layers=1, hid=32, k=3, k_in=4, loss="mse", flat=False,
                     fig="result_baseline.png", name="基线 Baseline"),
    "exp1": dict(layers=2, hid=32, k=3, k_in=4, loss="mse", flat=False,
                 fig="result_exp1.png", name="加深层数(2层)"),
    "exp2": dict(layers=1, hid=64, k=3, k_in=4, loss="mse", flat=False,
                 fig="result_exp2.png", name="隐藏通道64"),
    "exp3": dict(layers=1, hid=32, k=5, k_in=4, loss="mse", flat=False,
                 fig="result_exp3.png", name="卷积核K=5"),
    "exp4": dict(layers=1, hid=32, k=3, k_in=8, loss="mse", flat=False,
                 fig="result_exp4.png", name="看8帧"),
    "exp5": dict(layers=1, hid=32, k=3, k_in=4, loss="mse", flat=True,
                 fig="result_exp5.png", name="全连接FlattenLSTM"),
    "exp6": dict(layers=1, hid=32, k=3, k_in=4, loss="l1", flat=False,
                 fig="result_exp6.png", name="L1损失"),
}


def build_model(cfg):
    if cfg["flat"]:
        return FlattenLSTM()
    return ConvLSTM(in_ch=1, hid=cfg["hid"], k=cfg["k"], layers=cfg["layers"])


def run_group(group, seed=42, verbose=True):
    cfg = GROUP_CONFIG[group]
    train_x, train_y, test_x, test_y = build_dataset(k_in=cfg["k_in"], seed=42)
    model = build_model(cfg)
    hist, dt, mse, mae, params, model = train_one(
        model, train_x, train_y, test_x, test_y,
        loss_name=cfg["loss"], seed=seed)
    out_png = os.path.join(FIG_DIR, cfg["fig"])
    plot_curve(hist, f"{cfg['name']} 训练损失与测试MSE", out_png)
    rec = dict(group=group, name=cfg["name"], seed=seed,
               mse=round(mse, 6), mae=round(mae, 6),
               time_s=round(dt, 1), params=params,
               fig=cfg["fig"], k_in=cfg["k_in"],
               layers=cfg["layers"], hid=cfg["hid"], k=cfg["k"],
               loss=cfg["loss"])
    if verbose:
        print(f"[{group}] {cfg['name']}: MSE={mse:.5f} MAE={mae:.5f} "
              f"time={dt:.1f}s params={params}")
    return rec, model, (train_x, train_y, test_x, test_y)


def run_best(seed):
    """最优组合：综合两项经单独验证有效的改进——2层 + K=5（C_h=32，看4帧，MSE）。
    依据：exp1两层 MSE=0.00379、exp3 K5 MSE=0.00360，均优于基线0.00456；
    hid64(0.00423)与看8帧(0.00465)对MSE无明显改善，故不纳入。"""
    cfg = dict(layers=2, hid=32, k=5, k_in=4, loss="mse", flat=False)
    train_x, train_y, test_x, test_y = build_dataset(k_in=4, seed=42)
    model = ConvLSTM(in_ch=1, hid=32, k=5, layers=2)
    hist, dt, mse, mae, params, model = train_one(
        model, train_x, train_y, test_x, test_y, loss_name="mse", seed=seed)
    out_png = os.path.join(FIG_DIR, "result_best.png")
    plot_curve(hist, f"最优组合(2层/K=5) 种子{seed}", out_png)
    rec = dict(group="best", seed=seed, mse=round(mse, 6), mae=round(mae, 6),
               time_s=round(dt, 1), params=params)
    print(f"[best seed={seed}] MSE={mse:.5f} MAE={mae:.5f} time={dt:.1f}s params={params}")
    return rec, model, (train_x, train_y, test_x, test_y)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="baseline")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    if args.group == "best":
        recs = []
        model0, ds0 = None, None
        for s in [42, 43, 44][:args.trials]:
            rec, model, ds = run_best(s)
            recs.append(rec)
            if s == 42:
                model0, ds0 = model, ds
        # 用 seed=42 这次的模型生成 pred 对比图
        plot_pred(model0, ds0[2], ds0[3], os.path.join(FIG_DIR, "result_pred.png"))
        mse_avg = float(np.mean([r["mse"] for r in recs]))
        mae_avg = float(np.mean([r["mae"] for r in recs]))
        t_avg = float(np.mean([r["time_s"] for r in recs]))
        out = dict(trials=recs, avg_mse=round(mse_avg, 6),
                   avg_mae=round(mae_avg, 6), avg_time_s=round(t_avg, 1))
        with open(os.path.join(METRIC_DIR, "best.json"), "w") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print("BEST AVG:", out)
    else:
        rec, _, _ = run_group(args.group, seed=args.seed)
        with open(os.path.join(METRIC_DIR, f"{args.group}.json"), "w") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)
