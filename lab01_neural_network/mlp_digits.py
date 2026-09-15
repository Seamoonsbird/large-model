# -*- coding: utf-8 -*-
"""
================================================================================
实验作业一：使用 TRAE IDE 设计更好的神经网络
================================================================================
对应教材：第 2 章《神经元与神经网络》
数据集  ：sklearn 内置手写数字 load_digits（8×8 灰度图，1797 个样本，10 类）

任务要求（摘自实验指南）：
  1. 从零构建一个 MLP 分类器（禁止调用 sklearn 的 MLPClassifier）；
  2. 先建立"朴素基线（Baseline）"，再围绕激活函数 / 网络深度宽度 / 优化器 /
     学习率 / L2 正则化 / Dropout / BatchNorm 7 个维度开展 8 组对照实验；
  3. 全部实验共用同一份数据划分（训练 80% / 测试 20%，分层抽样，随机种子 42）
     与随机种子，每组只改一个变量（控制变量法）；
  4. 每组实验记录测试准确率与训练耗时，并保存损失 / 准确率曲线图；
  5. 必做一次"失败实验"（学习率过大 lr=10.0），分析不收敛原因；
  6. 把各维度中验证有效的改进组合成"最优组合"，固定种子复测 3 次取平均。

运行方式：
  python mlp_digits.py

运行结束后会在当前目录生成：
  result_baseline.png  result_exp1.png ... result_exp8_run1.png
  result_lr_too_big.png  result_lr_scan.png
并在终端打印所有实验的测试准确率与训练耗时汇总表。
================================================================================
"""

import time

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------------------------
# 0. 全局设置：中文字体 + 随机种子（保证实验可复现）
# ------------------------------------------------------------------------------
# Windows 用 Microsoft YaHei，macOS 可改为 ["Arial Unicode MS"]，Linux 可用
# Noto Sans CJK SC / SimHei 等，保证图上的中文不显示成方框。
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Microsoft YaHei",
                                   "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号

# 全局随机种子：固定后数据划分、模型参数初始化、Dropout 等全部可复现
torch.manual_seed(42)
np.random.seed(42)


# ------------------------------------------------------------------------------
# 1. 数据准备：加载 load_digits，8:2 分层划分
# ------------------------------------------------------------------------------
def load_data():
    """加载手写数字数据集并做 8:2 分层划分。

    返回：
        X_tr, X_te : 训练 / 测试特征（torch.float32，已归一化到 [0,1]）
        y_tr, y_te : 训练 / 测试标签（torch.long）
    说明：
        - load_digits 的像素值范围是 [0, 16]，除以 16 归一化到 [0, 1]，
          避免特征量级过大影响梯度；
        - stratify=y 表示分层抽样，保证训练 / 测试集中 0-9 各类比例一致；
        - 标签必须转成 torch.long：CrossEntropyLoss 要求 Long 类型，
          否则会报 "expected scalar type Long but found Int"（Windows 常见）。
    """
    X, y = load_digits(return_X_y=True)
    X = X.astype(np.float32) / 16.0

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    return (torch.tensor(X_tr), torch.tensor(y_tr, dtype=torch.long),
            torch.tensor(X_te), torch.tensor(y_te, dtype=torch.long))


# ------------------------------------------------------------------------------
# 2. 模型定义：可通过参数配置的 MLP
# ------------------------------------------------------------------------------
class MLP(nn.Module):
    """多层感知器（MLP），隐藏层结构、激活函数、Dropout、BatchNorm 均可配置。

    网络结构示意（以 hidden=(32,) 为例）：
        输入(64) → Linear(64→32) → [BatchNorm] → 激活 → [Dropout]
        → Linear(32→10) → 输出(10)

    参数：
        hidden   : 元组，每个元素是一个隐藏层的神经元数，如 (32,) 或 (128, 128)
        act      : 激活函数名，可选 "sigmoid" / "tanh" / "relu" / "gelu"
        dropout  : Dropout 概率，0 表示不使用
        use_bn   : 是否在每个隐藏层后插入 BatchNorm1d
    """

    def __init__(self, hidden=(32,), act="sigmoid", dropout=0.0, use_bn=False):
        super().__init__()
        # 激活函数字典：把字符串映射成对应的 PyTorch 层类
        act_layer = {
            "sigmoid": nn.Sigmoid,
            "tanh": nn.Tanh,
            "relu": nn.ReLU,
            "gelu": nn.GELU,
        }[act]

        layers, prev = [], 64  # 输入维度固定为 64（8×8 像素展平）
        for h in hidden:
            layers.append(nn.Linear(prev, h))     # 全连接层：加权和 + 偏置
            if use_bn:                            # 改进点①：BatchNorm
                layers.append(nn.BatchNorm1d(h))  #   逐层标准化，稳定训练
            layers.append(act_layer())            # 激活函数：引入非线性
            if dropout > 0:                       # 改进点②：Dropout
                layers.append(nn.Dropout(dropout))  # 随机失活，增强泛化
            prev = h
        layers.append(nn.Linear(prev, 10))        # 输出层：10 类
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """前向传播：把输入依次经过各层，返回未归一化的 logits。"""
        return self.net(x)


# ------------------------------------------------------------------------------
# 3. 训练与评估函数
# ------------------------------------------------------------------------------
def run(model, opt_name="sgd", lr=0.1, epochs=30, wd=0.0, seed=42):
    """用指定配置训练一个模型，返回历史曲线与训练耗时。

    训练三步循环（教材 4.2 节）：
        ① 前向传播   : pred = model(X_tr)，得到预测 logits
        ② 损失计算   : CrossEntropyLoss 量化预测与真实标签的误差
        ③ 反向传播   : loss.backward() 按链式法则求梯度
           参数更新   : opt.step() 沿负梯度方向更新参数 w ← w - η·∂L/∂w

    参数：
        model  : MLP 实例
        opt_name : "sgd"（可带动量，见下方注释）或 "adam"   —— 改进点③：优化器可切换
        lr     : 学习率
        epochs : 训练轮数（全批量，每轮一次参数更新）
        wd     : weight_decay，即 L2 正则化系数，0 表示不使用
        seed   : 本次运行的随机种子（用于最优组合复测）
    返回：
        hist : dict，含 "loss"（每轮训练损失）与 "acc"（每轮测试准确率）两个列表
        dt   : 训练总耗时（秒）
    """
    # 优化器选择：SGD 可选带动量，Adam 为自适应学习率优化器
    if opt_name == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "sgd_momentum":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                              weight_decay=wd)
    else:  # "adam"
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    loss_fn = nn.CrossEntropyLoss()          # 多分类交叉熵损失
    hist = {"loss": [], "acc": []}           # 记录每轮的训练损失与测试准确率

    t0 = time.time()
    for ep in range(epochs):
        # ---------- 训练模式 ----------
        model.train()                        # 启用 Dropout / BatchNorm 训练行为
        opt.zero_grad()                      # 清空上一步的梯度
        pred = model(X_tr)                   # ① 前向传播
        loss = loss_fn(pred, y_tr)           # ② 损失计算
        loss.backward()                      # ③ 反向传播，算出各参数梯度
        opt.step()                           # ④ 优化器沿负梯度更新参数

        # ---------- 评估模式 ----------
        model.eval()                         # 关闭 Dropout，用 BatchNorm 的全局统计量
        with torch.no_grad():                # 评估阶段不需要计算梯度，省内存
            acc = (model(X_te).argmax(dim=1) == y_te).float().mean().item()
        hist["loss"].append(loss.item())
        hist["acc"].append(acc)
    dt = time.time() - t0
    return hist, dt


# ------------------------------------------------------------------------------
# 4. 画图工具：损失曲线 + 准确率曲线双图
# ------------------------------------------------------------------------------
def plot_curves(hist, title, save_path):
    """绘制训练损失与测试准确率两条曲线并保存。

    参数：
        hist      : run() 返回的历史字典
        title     : 图题（会写在图上）
        save_path : 保存的图片文件名
    """
    epochs = range(1, len(hist["loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))

    # 左图：训练损失曲线
    ax1.plot(epochs, hist["loss"], "b-o", markersize=3)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("训练损失（CrossEntropy）")
    ax1.set_title("训练损失曲线")
    ax1.grid(alpha=0.3)

    # 右图：测试准确率曲线
    ax2.plot(epochs, np.array(hist["acc"]) * 100, "r-o", markersize=3)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("测试准确率（%）")
    ax2.set_title("测试准确率曲线")
    ax2.set_ylim(0, 105)
    ax2.grid(alpha=0.3)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(save_path, dpi=120)
    plt.close(fig)   # 及时关闭，避免占用内存
    print(f"    [图] 已保存: {save_path}")


# ------------------------------------------------------------------------------
# 5. 汇总输出：实验结果记录表
# ------------------------------------------------------------------------------
def print_summary(rows):
    """把各组实验结果按表格形式打印到终端，便于抄入报告表 3。"""
    print("\n" + "=" * 92)
    print("实验汇总表（对照实验设计与结果记录）")
    print("=" * 92)
    print(f"{'组号':<5}{'实验名称':<14}{'改动内容':<30}{'测试准确率':<13}{'训练耗时'}")
    print("-" * 88)
    for row in rows:
        print(f"{row['id']:<6}{row['name']:<15}{row['desc']:<31}"
              f"{row['acc'] * 100:>7.2f}%{row['time']:>10.2f}s")
    print("=" * 88)


# ==============================================================================
# 6. 主程序：按控制变量法依次运行所有实验
# ==============================================================================
if __name__ == "__main__":
    # ---------- 6.1 数据（全局只切一次，所有实验共用同一份划分） ----------
    X_tr, y_tr, X_te, y_te = load_data()
    print(f"数据就绪：训练集 {X_tr.shape[0]} 个样本，测试集 {X_te.shape[0]} 个样本")

    # ---------- 6.2 定义实验组（每组只改一个变量，其余与基线一致） ----------
    # 基线配置：64 → 1×32(Sigmoid) → 10，SGD(lr=0.1)，30 epoch，无正则化
    experiments = [
        {"id": "0", "name": "基线Baseline", "desc": "1×32 Sigmoid + SGD(0.1)",
         "hidden": (32,), "act": "sigmoid", "opt": "sgd", "lr": 0.1,
         "wd": 0.0, "dropout": 0.0, "bn": False, "save": "result_baseline.png"},
        {"id": "1", "name": "换激活函数", "desc": "Sigmoid → ReLU",
         "hidden": (32,), "act": "relu", "opt": "sgd", "lr": 0.1,
         "wd": 0.0, "dropout": 0.0, "bn": False, "save": "result_exp1.png"},
        {"id": "2", "name": "加深网络", "desc": "1层 → 2层×128神经元",
         "hidden": (128, 128), "act": "sigmoid", "opt": "sgd", "lr": 0.1,
         "wd": 0.0, "dropout": 0.0, "bn": False, "save": "result_exp2.png"},
        {"id": "3", "name": "换优化器", "desc": "SGD → Adam(0.01)",
         "hidden": (32,), "act": "sigmoid", "opt": "adam", "lr": 0.01,
         "wd": 0.0, "dropout": 0.0, "bn": False, "save": "result_exp3.png"},
        {"id": "4", "name": "调学习率", "desc": "0.1 → 0.01",
         "hidden": (32,), "act": "sigmoid", "opt": "sgd", "lr": 0.01,
         "wd": 0.0, "dropout": 0.0, "bn": False, "save": "result_exp4.png"},
        {"id": "5", "name": "加L2正则", "desc": "weight_decay = 1e-4",
         "hidden": (32,), "act": "sigmoid", "opt": "sgd", "lr": 0.1,
         "wd": 1e-4, "dropout": 0.0, "bn": False, "save": "result_exp5.png"},
        {"id": "6", "name": "加Dropout", "desc": "p = 0.2",
         "hidden": (32,), "act": "sigmoid", "opt": "sgd", "lr": 0.1,
         "wd": 0.0, "dropout": 0.2, "bn": False, "save": "result_exp6.png"},
        {"id": "7", "name": "加BatchNorm", "desc": "隐藏层后插BatchNorm1d",
         "hidden": (32,), "act": "sigmoid", "opt": "sgd", "lr": 0.1,
         "wd": 0.0, "dropout": 0.0, "bn": True, "save": "result_exp7.png"},
        {"id": "8", "name": "最优组合", "desc": "ReLU+2×128+BN+Adam(0.001)",
         "hidden": (128, 128), "act": "relu", "opt": "adam", "lr": 0.001,
         "wd": 0.0, "dropout": 0.0, "bn": True, "save": "result_exp8_run1.png"},
    ]

    # ---------- 6.3 依次运行 0~8 组实验 ----------
    rows = []                      # 保存各组结果，最后打印汇总表
    for exp in experiments:
        print(f"\n>>> 实验 {exp['id']}：{exp['name']}  （{exp['desc']}）")
        # 每组实验开始前重新固定随机种子，保证参数初始化一致（控制变量）
        torch.manual_seed(42)
        model = MLP(hidden=exp["hidden"], act=exp["act"],
                    dropout=exp["dropout"], use_bn=exp["bn"])
        hist, dt = run(model, opt_name=exp["opt"], lr=exp["lr"],
                       epochs=30, wd=exp["wd"])
        acc = hist["acc"][-1]      # 取最后一轮的测试准确率
        print(f"    测试准确率: {acc * 100:.2f}%   训练耗时: {dt:.2f}s")
        # 实验 8（最优组合）的曲线图在 6.6 节复测时统一保存，这里不再重复画
        if exp["id"] != "8":
            plot_curves(hist, f"实验 {exp['id']}：{exp['name']}（{exp['desc']}）",
                        exp["save"])
        rows.append({"id": exp["id"], "name": exp["name"], "desc": exp["desc"],
                     "acc": acc, "time": dt})

    # ---------- 6.4 失败实验：学习率过大（lr = 10.0），不计入改进组 ----------
    print("\n>>> 失败实验：学习率过大 lr=10.0（基线其余配置不变）")
    torch.manual_seed(42)
    model_bad = MLP(hidden=(32,), act="sigmoid")
    hist_bad, dt_bad = run(model_bad, opt_name="sgd", lr=10.0, epochs=30)
    print(f"    测试准确率: {hist_bad['acc'][-1] * 100:.2f}%   训练耗时: {dt_bad:.2f}s")
    plot_curves(hist_bad, "失败实验：学习率过大（lr=10.0）导致不收敛",
                "result_lr_too_big.png")

    # ---------- 6.5 附加观察（加分项）：扫描学习率，寻找发散临界点 ----------
    print("\n>>> 附加观察：学习率扫描 0.1 / 1.0 / 5.0 / 10.0 / 20.0")
    lr_scan = [0.1, 1.0, 5.0, 10.0, 20.0]
    scan_accs = []
    for lr in lr_scan:
        torch.manual_seed(42)
        m = MLP(hidden=(32,), act="sigmoid")
        h, _ = run(m, opt_name="sgd", lr=lr, epochs=30)
        scan_accs.append(h["acc"][-1])
        print(f"    lr={lr:<5} 测试准确率: {h['acc'][-1] * 100:.2f}%")
    # 画一张学习率扫描对比图，直观展示"发散临界点"
    plt.figure(figsize=(7, 4.2))
    plt.plot(lr_scan, np.array(scan_accs) * 100, "o-", color="darkorange")
    plt.xscale("log")
    plt.xlabel("学习率（对数坐标）")
    plt.ylabel("测试准确率（%）")
    plt.title("学习率扫描：过大学习率导致性能崩塌")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("result_lr_scan.png", dpi=120)
    plt.close()
    print("    [图] 已保存: result_lr_scan.png")

    # ---------- 6.6 最优组合复测：固定种子跑 3 次取平均 ----------
    print("\n>>> 最优组合复测：固定随机种子（42/2026/7）运行 3 次取平均")
    best_accs, best_times = [], []
    for i, s in enumerate([42, 2026, 7], start=1):
        torch.manual_seed(s)
        model_best = MLP(hidden=(128, 128), act="relu", use_bn=True)
        hb, tb = run(model_best, opt_name="adam", lr=0.001, epochs=30)
        best_accs.append(hb["acc"][-1])
        best_times.append(tb)
        print(f"    第 {i} 次（seed={s}）: 测试准确率 {hb['acc'][-1] * 100:.2f}%"
              f"   耗时 {tb:.2f}s")
        if i == 1:  # 第一次运行的结果图用作报告图 9
            plot_curves(hb, "最优组合：ReLU + 2×128 + BatchNorm + Adam(0.001)",
                        "result_exp8_run1.png")
    avg_acc = float(np.mean(best_accs))
    avg_time = float(np.mean(best_times))
    print(f"    平均测试准确率: {avg_acc * 100:.2f}%   平均耗时: {avg_time:.2f}s")

    # ---------- 6.7 打印实验汇总表（用于填写报告表 3） ----------
    rows.append({"id": "—", "name": "失败实验", "desc": "lr=10.0（Sigmoid+SGD）",
                 "acc": hist_bad["acc"][-1], "time": dt_bad})
    print_summary(rows)

    # ---------- 6.8 最终结论输出 ----------
    print("\n实验全部完成！以上准确率与耗时为真实运行结果，可直接填入报告。")
