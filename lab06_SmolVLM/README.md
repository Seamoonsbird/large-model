# 实验六 · SmolVLM-256M 精准定位的 RSI 递归自进化

## 环境
- Python 3.10+ / PyTorch 2.2+（CPU 即可）
- transformers 4.44+ / peft 0.11+ / Pillow
- 安装：`pip install torch transformers peft accelerate pillow`
- 国内下载模型建议：`export HF_ENDPOINT=https://hf-mirror.com`

## 运行顺序
```bash
python hw6_rsi_localize.py --mode make_data       # 生成 40 训练图 + 15 评测图 + gt.json
python hw6_rsi_localize.py --mode make_sft        # 由 GT 生成 15 条种子 SFT 数据
python hw6_rsi_localize.py --mode infer           # 基线推理（训练前）
python hw6_rsi_localize.py --mode sft --steps 10   # LoRA 微调 10 步
python hw6_rsi_localize.py --mode eval             # 三阶段评测：基线 / SFT / SFT+纠错
python hw6_rsi_localize.py --mode rsi --rounds 2  # RSI 两轮递归自进化
```

## 产物
- `hw6_imgs/`：合成数据集（train/eval/gt.json）
- `sft_data.json`：SFT 数据池（随 RSI 轮次增长）
- `sft_adapter/`：LoRA 权重
- `rsi_memory.json`：RSI 每轮自纠错记忆
