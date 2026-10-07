# -*- coding: utf-8 -*-
"""
任务五(b) 增量更新演示：在 CORPUS 中新增 doc16，评估三种范式各需要改动什么、约花几分钟。
用法：python hw4_doc16_demo.py
（独立演示脚本，不参与 hw4_rag.py 的主评测；主评测仍用原始 15 篇语料，保证结果可复现。）
"""
import importlib.util, time, re, zlib
import numpy as np

spec = importlib.util.spec_from_file_location("hw4", "hw4_rag.py")
hw4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hw4)

DOC16 = ("doc16", "人工智能学院新增教师：韩雨桐，副教授，2025年入职，研究方向为自然语言处理，主讲《自然语言处理导论》（课程代码AI4402）。")

def fmt(t):
    return f"{t*1000:.1f} ms" if t < 1 else f"{t:.2f} s"

print("=" * 72)
print(f"新增文档 doc16：{DOC16[1]}")
print("=" * 72)

emb = hw4.Embedder("paraphrase-multilingual-MiniLM-L12-v2")

# ---------- 1. Vector RAG ----------
print("\n【Vector RAG】")
t0 = time.time()
chunks_all = hw4.chunk_docs()
t_build_base = time.time() - t0
print(f"  ① 初始构建：分块 {len(chunks_all)} 块，全量嵌入 {fmt(t_build_base)}")

# 增量：只对 doc16 分块并嵌入，追加到向量矩阵
t0 = time.time()
new_chunks = hw4.chunk_docs()  # 实际仅需 doc16 的块；这里模拟“只处理新文档”
doc16_chunks = [(d, t) for d, t in chunks_all if d == "doc16"]
if not doc16_chunks:  # 若原语料无 doc16，则单独分块
    text = DOC16[1]
    size, overlap = 120, 20
    doc16_chunks = []
    i = 0
    while True:
        doc16_chunks.append((DOC16[0], text[i:i + size]))
        if i + size >= len(text):
            break
        i += size - overlap
new_vec = emb.encode([c[1] for c in doc16_chunks])
t_inc = time.time() - t0
print(f"  ② 增量更新：新增 {len(doc16_chunks)} 个块（{fmt(t_inc)}），"
      f"旧 {len(chunks_all)} 个块向量不动，矩阵追加一行即可")
print(f"  ③ 需改动：chunk_docs() 的输入 CORPUS 增加 doc16 → 向量矩阵 M 追加 1 行（无需重算旧块）")

# ---------- 2. GraphRAG ----------
print("\n【GraphRAG】")
import networkx as nx
G = nx.DiGraph()
for h, r, t, src in hw4.PRE_TRIPLES:
    G.add_edge(h, t, rel=r, src=src)
t0 = time.time()
comms = nx.community.greedy_modularity_communities(G.to_undirected())
t_base = time.time() - t0
print(f"  ① 初始构建：实体 {G.number_of_nodes()} 个，社区发现 {fmt(t_base)}")

# 增量：新增 3 条三元组（韩雨桐 相关），重跑社区发现
new_triples = [
    ("韩雨桐", "任职于", "人工智能学院", "doc16"),
    ("韩雨桐", "研究方向", "自然语言处理", "doc16"),
    ("韩雨桐", "主讲", "自然语言处理导论", "doc16"),
]
t0 = time.time()
for h, r, t, src in new_triples:
    G.add_edge(h, t, rel=r, src=src)
t_edge = time.time() - t0
t0 = time.time()
comms2 = nx.community.greedy_modularity_communities(G.to_undirected())
t_comm = time.time() - t0
print(f"  ② 增量更新：新增 3 条边（{fmt(t_edge)}）+ 重跑社区发现（{fmt(t_comm)}）")
print(f"  ③ 需改动：PRE_TRIPLES 追加 3 条 → 图加边；社区成员变化后，"
      f"C1 社区摘要需由 LLM 重写（教材8.7节，预置版需人工/LLM 补充『韩雨桐…自然语言处理导论』）")

# ---------- 3. WikiRAG ----------
print("\n【WikiRAG】")
t0 = time.time()
base_vec = emb.encode([b for _, b, _ in hw4.PRE_ENTRIES])
t_base = time.time() - t0
print(f"  ① 初始构建：9 个条目全量嵌入 {fmt(t_base)}")

# 增量：把新教师并入「人工智能学院教师」条目，仅重嵌入该条目
t0 = time.time()
entry_idx = [i for i, e in enumerate(hw4.PRE_ENTRIES) if e[0] == "人工智能学院教师"][0]
old_entry = hw4.PRE_ENTRIES[entry_idx]
new_body = old_entry[1] + "韩雨桐：副教授，2025年入职，研究自然语言处理，主讲《自然语言处理导论》。"
new_vec = emb.encode([new_body])
t_inc = time.time() - t0
print(f"  ② 增量更新：合并写入条目「{old_entry[0]}」（{fmt(t_inc)}），仅重嵌入 1 行")
print(f"  ③ 需改动：PRE_ENTRIES 中『人工智能学院教师』正文追加韩雨桐 → 条目向量矩阵该行更新；"
      f"其余 8 个条目不动")

print("\n" + "=" * 72)
print("结论（三种范式更新成本对比）：")
print(f"  Vector RAG：只加块、只嵌新块，约 {fmt(t_inc)}（最轻，无需改旧数据）")
print(f"  GraphRAG ：加 3 条边 + 重跑社区发现（{fmt(t_comm)}）+ 重写 1 条社区摘要（LLM 成本，预置版为人工）")
print(f"  WikiRAG  ：改 1 个条目的正文并重嵌入（{fmt(t_inc)}），条目化维护成本中等")
print("=" * 72)
