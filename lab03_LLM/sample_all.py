# -*- coding: utf-8 -*-
"""加载已训练权重，批量生成采样样本（含温度/top-k对照）。"""
import os, torch
import torch.nn.functional as F
from min_llm import build_corpus, CharTokenizer, MiniGPT

os.chdir(os.path.dirname(os.path.abspath(__file__)))
text = build_corpus("corpus_extra.txt")
tok = CharTokenizer(text)
print(f"V={tok.vocab_size}")

def load_model(ckpt, n_embed, n_head, n_layer, block_size, use_pos):
    m = MiniGPT(tok.vocab_size, n_embed, n_head, n_layer, block_size, use_pos=use_pos)
    sd = torch.load(ckpt, map_location="cpu")["model_state"]
    m.load_state_dict(sd); m.eval()
    return m

def gen(m, prompt, temp, top_k, n=1, ntok=120, seed=42):
    outs = []
    for s in range(n):
        torch.manual_seed(seed + s*13)
        idx = torch.tensor([tok.encode(prompt)], dtype=torch.long)
        out = m.generate(idx, ntok, temperature=temp, top_k=top_k)
        outs.append(tok.decode(out[0].tolist()))
    return outs

# ---------- 1) 基线模型：5 组采样对照 ----------
base = load_model("min_llm.pt", 128, 4, 2, 128, True)
groups = [
    ("s1_T1.0_k20",  1.0, 20),
    ("s2_T0.5_k20",  0.5, 20),
    ("s3_T1.5_k20",  1.5, 20),
    ("s4_T1.0_k5",   1.0, 5),
    ("s5_T1.0_k699", 1.0, 699),
]
with open("sampling_results.txt", "w", encoding="utf-8") as f:
    for tag, T, k in groups:
        f.write(f"===== {tag} (temperature={T}, top_k={k}) =====\n")
        for prompt in ["春", "月"]:
            for g in gen(base, prompt, T, k, n=2):
                f.write(f"[提示词「{prompt}」]\n{g}\n\n")
        f.write("\n")
print("sampling_results.txt done")

# ---------- 2) 重新生成 baseline 的 generated_base.txt ----------
with open("generated_base.txt", "w", encoding="utf-8") as f:
    for prompt in ["春", "月"]:
        for i, g in enumerate(gen(base, prompt, 1.0, 20, n=2)):
            f.write(f"【提示词「{prompt}」 第{i+1}段 | T=1.0 top_k=20】\n{g}\n\n")
print("generated_base.txt done")

# ---------- 3) 各对照实验模型的生成样本（用于质量评级） ----------
exps = [
    ("exp1a_lr0.01",   "exp1a.pt", 128, 4, 2, 128, True),
    ("exp1b_lr0.0001", "exp1b.pt", 128, 4, 2, 128, True),
    ("exp2a_L1",       "exp2a.pt", 128, 4, 1, 128, True),
    ("exp2b_L4",       "exp2b.pt", 128, 4, 4, 128, True),
    ("exp3a_E64",      "exp3a.pt", 64,  2, 2, 128, True),
    ("exp3b_E256",     "exp3b.pt", 256, 8, 2, 128, True),
    ("exp4_T32",       "exp4.pt",  128, 4, 2, 32,  True),
    ("exp5_nopos",     "exp5.pt",  128, 4, 2, 128, False),
]
with open("exp_generations.txt", "w", encoding="utf-8") as f:
    for tag, ck, E, H, L, T, pos in exps:
        m = load_model(ck, E, H, L, T, pos)
        f.write(f"===== {tag} =====\n")
        for prompt in ["春"]:
            for g in gen(m, prompt, 1.0, 20, n=2):
                f.write(f"[「{prompt}」]\n{g}\n\n")
print("exp_generations.txt done")
