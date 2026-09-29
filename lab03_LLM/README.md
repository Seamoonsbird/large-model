# 实验作业三：手搓最小 LLM —— 使用 CPU 训练

## 一、目录内容

| 文件 | 说明 |
|---|---|
| `min_llm.py` | 主程序：内置唐诗语料、字符级分词、最小 GPT、CPU 训练与自回归采样（单文件可直接运行） |
| `sample_all.py` | 加载已训练权重，批量生成 5 组温度/top-k 采样对照与各组实验生成样本 |
| `fill_report.py` | 本报告的自动填写脚本（供参考） |
| `学号_姓名_实验作业三.docx` | 填好的实验报告（请把文件名里的“学号_姓名”改成自己的） |
| `loss_curve_*.png` | 各组实验的训练 loss 曲线（共 9 张） |
| `generated_base.txt` | 基线模型用「春」「月」各生成 2 段（T=1.0, top_k=20） |
| `sampling_results.txt` | 5 组温度/top-k 采样完整原文 |
| `exp_generations.txt` | 各对照实验模型的生成样本（用于质量评级） |
| `*.pt` | 各实验训练好的模型权重 |
| `run_exp*.log / train_baseline.log` | 训练日志（含参数量、loss、耗时） |

## 二、复现方法

```bash
# 1. 安装依赖（CPU 版即可）
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install matplotlib

# 2. 基线训练（2000 步，约 4~25 分钟，视 CPU 而定）
python min_llm.py --iters 2000

# 3. 快速自检
python min_llm.py --iters 800

# 4. 对照实验示例
python min_llm.py --iters 1000 --lr 1e-2        # exp1a
python min_llm.py --iters 1000 --n_layer 1       # exp2a
python min_llm.py --iters 1000 --n_embed 64 --n_head 2   # exp3a
python min_llm.py --iters 1000 --block_size 32   # exp4
python min_llm.py --iters 1000 --no_pos          # exp5

# 5. 加载基线权重做采样对照
python sample_all.py
```

## 三、关键结果速览

- 词表 V=699，参数量 **502,656**（手算 499,072，误差 0.71%<1%）
- 初始 loss 6.56 ≈ ln 699 ≈ 6.55；基线最终 loss 0.0207，耗时 4.0 min
- 6 组对照实验的最终 loss、生成质量星级与现象见报告表 5
