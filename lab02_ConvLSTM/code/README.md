# 作业二 ConvLSTM 算法应用与改进 — 代码运行说明

## 目录结构
```
chats/38441736218918402/
├── code/
│   ├── run_exp.py        # 全部实验主脚本（数据/模型/训练/绘图/评估）
│   └── run_all.sh        # 组0~组6 并行驱动（2核，每次2组）
├── figures/              # 结果图（≥150dpi，中文用 Noto Sans CJK SC）
│   ├── result_baseline.png / result_exp1~6.png / result_best.png
│   ├── result_pred.png    # 最优组合"输入|真实|预测"九宫格
│   └── param_check.png    # 参数量 sum(p.numel()) 打印截图
└── metrics/
    ├── baseline.json / exp1~6.json   # 各组 {mse, mae, time_s, params,...}
    ├── metrics.json / metrics.csv    # 汇总
    └── best.json                     # 最优组合3次复测与平均
```

## 环境
- Linux, Python 3.12, PyTorch 2.14.0+CPU, NumPy/Matplotlib
- 中文字体：/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc（无 Microsoft YaHei，已自动注册）

## 一键复现
```bash
cd code
# 组0~组6（2核并行，约25分钟）
bash run_all.sh
# 最优组合，固定种子 42/43/44 各跑一次取平均，并生成 result_pred.png
python3 run_exp.py --group best --trials 3 --threads 2
# 单独跑某一组
python3 run_exp.py --group baseline --seed 42 --threads 2
```

## 关键设定（与指南一致，并修复参考代码 bug）
- 数据：弹跳小球，T=10、32×32、r=2、初速度0.8~1.6、边界反弹、种子42；
  **n_seq=2200，train=data[:2000]、test=data[2000:]，互不重叠**（修正参考代码 test=data[1800:] 与训练重叠的 bug）。
- 基线：看前4帧预测第5帧；ConvLSTMCell 单 Conv2d(in+hid,4*hid,k,padding=k//2)，
  z.chunk(4) 分 i/f/g/o 四门，c_t=f*c+i*tanh(g)，h_t=o*tanh(c)；
  C_in=1,C_h=32,K=3,1层；输出 Conv2d(32→1,3×3,padding=1)；MSE+Adam(0.001)；batch64；5epoch；种子42。
- 修正了参考代码中 return 缩进在时间循环内、`if __name__=="__name__"` 两处 bug。

## 最优组合选取依据
各单独改动相对基线(MSE=0.00456)：
- exp1 两层 = 0.00379 ✓有效；exp3 K=5 = 0.00360 ✓有效；
- exp2 C_h64 = 0.00423（略降但MAE反升，收益有限）；exp4 看8帧 = 0.00465（MSE无改善）；
- exp5 展平LSTM = 0.00726（明显更差）；exp6 L1 = MSE 0.01187（MSE更差、MAE略好）。
故最优组合 = **2层 + K=5**（仅纳入经验证有效的两项），固定种子 42/43/44 跑3次取平均。
