# -*- coding: utf-8 -*-
"""
Vector RAG / GraphRAG / WikiRAG 对比实验 — 实验作业四（hw4_rag.py）
对应教材第8章《检索增强生成》
单文件、开箱即跑：内置「云山大学」语料15篇、标注问题15道（另自拟5题，共20题）、
预抽取三元组40条与条目库9条。不配LLM API也能完成全部检索评测；
配置API后可用 --mode vector --q Q01 体验完整「检索 + 生成」。

常用命令：
python hw4_rag.py --mode eval                  # 三范式 × 全部题 检索评测（无需API）
python hw4_rag.py --mode eval --subset builtin # 仅内置15题（K敏感性扫描用）
python hw4_rag.py --mode eval --detail         # 逐题打印检索结果（失败案例/手算用）
python hw4_rag.py --mode vector --q Q01        # 单题：检索 + 生成
python hw4_rag.py --mode graph --q Q01 --scope local
python hw4_rag.py --mode wiki --q Q11
python hw4_rag.py --mode extract               # 调LLM现场抽三元组（需API）
python hw4_rag.py --mrr_check 3,1,2            # MRR 手算复核（对照式8-10）
"""

import argparse, json, os, re, sys, time, zlib

# ==============================================
# 1. 语料库「云山大学」：15篇虚构文档（三类问题的答案均已埋入）
# ==============================================
CORPUS = {
    "doc01": "云山大学位于岭南省云州市，建于1958年，是省属重点大学，现有全日制本科生约2.1万人。校训为「格物致知」。",
    "doc02": "人工智能学院成立于2018年，首任及现任院长为王启明教授。学院下设机器学习系、智能科学系、认知计算系三个系。",
    "doc03": "计算机学院前身为1985年成立的计算机系，现任院长陈国峰教授，设计算机科学与技术、软件工程两个本科专业。",
    "doc04": "人工智能学院教师：李文瀚，教授，2019年入职，研究方向为检索增强生成，主讲《大模型通识课》；赵婉晴，副教授，研究方向为知识图谱，主讲《知识图谱导论》；孙浩然，讲师，研究方向为强化学习。",
    "doc05": "计算机学院教师：周天宇，教授，研究方向为分布式系统；林小雨，副教授，研究方向为数据库系统。",
    "doc06": "《大模型通识课》课程代码AI2101，3学分、32学时，春季学期开设，授课教师李文瀚，面向全校本科生，无先修课程要求。",
    "doc07": "《知识图谱导论》课程代码AI3305，3学分，秋季学期开设，授课教师赵婉晴，面向人工智能学院研究生。",
    "doc08": "选课规则：本科生每学期最多修30学分；GPA低于2.0给予学术警告。先修要求：《知识图谱导论》需先修《数据结构》，《大模型通识课》无先修要求。",
    "doc09": "奖学金体系：国家奖学金8000元/年，评定比例约2%；校长奖学金20000元/年，全校每年10人；云山一等奖学金3000元/年，比例约5%。",
    "doc10": "科研平台：认知计算全国重点实验室依托人工智能学院建设；岭南超算中心由云山大学与省科技厅共建，计算机学院参与运行管理。",
    "doc11": "学生社团：AI协会，指导教师李文瀚，每周三晚组织论文研读；机器人战队，指导教师孙浩然，每年参加全国机器人大赛。",
    "doc12": "校历：春季学期3月2日开学、7月5日放暑假；秋季学期9月1日开学、次年1月15日放寒假。",
    "doc13": "图书馆藏书380万册，开放时间为每天7:00-22:00；人工智能分馆位于理科楼B座3层，收藏大模型与智能体专题图书。",
    "doc14": "校园交通：地铁3号线「云大站」距东门200米；校内校车共5条线路，10分钟一班。",
    "doc15": "国际交流：学校与12个国家的47所高校签有交换协议；人工智能学院与新加坡南洋理工大学有本科联合培养项目。",
}

# ==============================================
# 2. 问题集：3类×5题内置 + 5题自拟：evidence = 支持答案的文档（评测金标准）
#    fact=局部事实  multi=多跳关系  global=全局归纳
# ==============================================
QUESTIONS = [
    # ---- 内置15题 ----
    {"id": "Q01", "type": "multi",  "q": "《大模型通识课》授课教师所在学院的行政负责人是谁？",
     "evidence": ["doc04", "doc06", "doc02"]},
    {"id": "Q02", "type": "multi",  "q": "AI协会的指导教师主讲课程的课程代码是什么？",
     "evidence": ["doc11", "doc04", "doc02"]},
    {"id": "Q03", "type": "multi",  "q": "认知计算全国重点实验室依托的学院成立于哪一年？",
     "evidence": ["doc10", "doc02"]},
    {"id": "Q04", "type": "multi",  "q": "《知识图谱导论》授课教师的研究方向是什么？",
     "evidence": ["doc07", "doc04"]},
    {"id": "Q05", "type": "multi",  "q": "机器人战队的指导教师的研究方向是什么？",
     "evidence": ["doc11", "doc04"]},
    {"id": "Q06", "type": "fact",   "q": "云山大学的校训是什么？", "evidence": ["doc01"]},
    {"id": "Q07", "type": "fact",   "q": "国家奖学金的金额是多少？", "evidence": ["doc09"]},
    {"id": "Q08", "type": "fact",   "q": "图书馆每天几点到几点开放？", "evidence": ["doc13"]},
    {"id": "Q09", "type": "fact",   "q": "人工智能学院成立于哪一年？", "evidence": ["doc02"]},
    {"id": "Q10", "type": "fact",   "q": "春季学期什么时候开学？", "evidence": ["doc12"]},
    {"id": "Q11", "type": "global", "q": "云山大学有哪些科研平台？分别依托谁建设？",
     "evidence": ["doc10", "doc02", "doc03"]},
    {"id": "Q12", "type": "global", "q": "学校的学生奖励体系包含哪些项目？", "evidence": ["doc09"]},
    {"id": "Q13", "type": "global", "q": "人工智能学院有哪些教师？各自做什么研究方向？",
     "evidence": ["doc04"]},
    {"id": "Q14", "type": "global", "q": "本科生选课有哪些限制和先修要求？", "evidence": ["doc08"]},
    {"id": "Q15", "type": "global", "q": "对想深入学习大模型的学生，学校提供哪些课程和课外活动？",
     "evidence": ["doc06", "doc11"]},
    # ---- 自拟5题（任务四①，每类至少1道；均标注evidence，避免与内置题雷同）----
    {"id": "Q16", "type": "fact",   "q": "《大模型通识课》的课程代码是什么？", "evidence": ["doc06"]},
    {"id": "Q17", "type": "fact",   "q": "地铁3号线「云大站」距离东门多远？", "evidence": ["doc14"]},
    {"id": "Q18", "type": "multi",  "q": "《知识图谱导论》授课教师所在学院的院长是谁？",
     "evidence": ["doc07", "doc04", "doc02"]},
    {"id": "Q19", "type": "multi",  "q": "AI协会的指导教师的研究方向是什么？",
     "evidence": ["doc11", "doc04"]},
    {"id": "Q20", "type": "global", "q": "学校在科研平台和国际合作方面有哪些布局？",
     "evidence": ["doc10", "doc15"]},
]

# ==============================================
# 3. 分块（教材8.3节：块长=300 token，中文按120字近似，重叠20字）
# ==============================================
def chunk_docs(size=120, overlap=20):
    chunks, step = [], size - overlap
    for doc_id, text in CORPUS.items():
        i = 0
        while True:
            chunks.append((doc_id, text[i:i + size]))
            if i + size >= len(text):
                break
            i += step
    return chunks

# ==============================================
# 4. 嵌入后端：优先 sentence-transformers（教材8.4节）；
#    未安装/无网络时自动降级为字符3-gram相似度（流程可跑通，精度有限）
# ==============================================
class Embedder:
    def __init__(self, model_name):
        self.ok = False
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.ok = True
            print(f"[embed] 已加载 {model_name}")
        except Exception as e:
            print(f"[embed] 加载失败 ({type(e).__name__})，降级为字符3-gram相似度（精度有限）")

    def encode(self, texts):
        import numpy as np
        if self.ok:
            return self.model.encode(list(texts), normalize_embeddings=True)
        M = np.zeros((len(texts), 512))
        for r, t in enumerate(texts):
            t = re.sub(r"\s", "", t)
            for i in range(max(1, len(t) - 2)):
                g = t[i:i + 3] if len(t) >= 3 else t
                M[r, zlib.crc32(g.encode("utf-8")) % 512] += 1
            n = np.linalg.norm(M[r]) or 1.0
            M[r] /= n
        return M

def _dedup(scored, k):
    docs, seen = [], set()
    for doc, s in scored:
        # scored 已按分数降序
        if doc not in seen:
            docs.append(doc)
            seen.add(doc)
            if len(docs) >= k:
                break
    return docs

# ==============================================
# 5. Vector RAG（教材8.2~8.4节）：块向量 → 余弦Top-K → 文档序列
# ==============================================
class VectorRAG:
    name = "Vector RAG"

    def __init__(self, emb):
        self.emb = emb
        self.chunks = chunk_docs()
        self.M = emb.encode([c[1] for c in self.chunks])

    def retrieve(self, query, k=5, scope=None):
        qv = self.emb.encode([query])[0]
        scores = self.M @ qv
        order = scores.argsort()[::-1]
        hits = [(self.chunks[i][0], float(scores[i])) for i in order]
        return _dedup(hits, k)

# ==============================================
# 6. GraphRAG（教材8.7节简化版）
#    预抽取三元组（头实体，关系，尾实体，来源文档；--mode extract可现场抽取）
# ==============================================
PRE_TRIPLES = [
    ("李文瀚", "任职于", "人工智能学院", "doc04"), ("李文瀚", "职称", "教授", "doc04"),
    ("李文瀚", "入职年份", "2019年", "doc04"), ("李文瀚", "研究方向", "检索增强生成", "doc04"),
    ("李文瀚", "主讲", "大模型通识课", "doc06"), ("李文瀚", "指导", "AI协会", "doc11"),
    ("赵婉晴", "任职于", "人工智能学院", "doc04"), ("赵婉晴", "研究方向", "知识图谱", "doc04"),
    ("赵婉晴", "主讲", "知识图谱导论", "doc07"), ("孙浩然", "任职于", "人工智能学院", "doc04"),
    ("孙浩然", "研究方向", "强化学习", "doc04"), ("孙浩然", "指导", "机器人战队", "doc11"),
    ("人工智能学院", "院长", "王启明", "doc02"), ("人工智能学院", "成立于", "2018年", "doc02"),
    ("人工智能学院", "下设系", "机器学习系", "doc02"), ("人工智能学院", "下设系", "智能科学系", "doc02"),
    ("人工智能学院", "下设系", "认知计算系", "doc02"), ("认知计算全国重点实验室", "依托", "人工智能学院", "doc10"),
    ("岭南超算中心", "参与共建", "计算机学院", "doc10"), ("计算机学院", "院长", "陈国峰", "doc03"),
    ("计算机学院", "前身", "计算机系", "doc03"), ("周天宇", "任职于", "计算机学院", "doc05"),
    ("周天宇", "研究方向", "分布式系统", "doc05"), ("林小雨", "任职于", "计算机学院", "doc05"),
    ("林小雨", "研究方向", "数据库系统", "doc05"), ("大模型通识课", "课程代码", "AI2101", "doc06"),
    ("大模型通识课", "学分", "3学分", "doc06"), ("大模型通识课", "开设学期", "春季学期", "doc06"),
    ("知识图谱导论", "课程代码", "AI3305", "doc07"), ("知识图谱导论", "先修课程", "数据结构", "doc08"),
    ("AI协会", "指导教师", "李文瀚", "doc11"), ("机器人战队", "指导教师", "孙浩然", "doc11"),
    ("国家奖学金", "金额", "8000元/年", "doc09"), ("校长奖学金", "金额", "20000元/年", "doc09"),
    ("云山一等奖学金", "金额", "3000元/年", "doc09"), ("云山大学", "校训", "格物致知", "doc01"),
    ("云山大学", "建校于", "1958年", "doc01"), ("云山大学", "位于", "岭南省云州市", "doc01"),
    ("图书馆", "藏书量", "380万册", "doc13"), ("春季学期", "开学日期", "3月2日", "doc12"),
]

# 社区摘要（教材8.7节由LLM生成；此处预置3条，对应图的三个稠密子区）
COMMUNITY_SUMMARIES = [
    ("社区C1 教学与课程：人工智能学院设三个系，开设《大模型通识课》（AI2101，李文瀚主讲）与《知识图谱导论》（AI3305，赵婉晴主讲），AI3305需先修《数据结构》。",
     ["doc02", "doc04", "doc06", "doc07", "doc08"]),
    ("社区C2 科研与国际：认知计算全国重点实验室依托人工智能学院，岭南超算中心由计算机学院参与共建，学院与新加坡南洋理工大学有联合培养项目。",
     ["doc02", "doc03", "doc10", "doc15"]),
    ("社区C3 学生生活：奖学金分国家（8000元）、校长（20000元）、云山一等（3000元）三级；社团有AI协会与机器人战队，图书馆藏书380万册、7:00-22:00开放。",
     ["doc09", "doc11", "doc13"]),
]

class GraphRAG:
    name = "GraphRAG"

    def __init__(self, emb):
        import networkx as nx
        self.emb = emb
        self.G = nx.DiGraph()
        for h, r, t, src in PRE_TRIPLES:
            self.G.add_edge(h, t, rel=r, src=src)
        try:  # 教材8.7节用Louvain；此处用networkx内置贪心模块社区发现
            comms = nx.community.greedy_modularity_communities(self.G.to_undirected())
        except Exception:
            comms = [set(self.G.nodes)]
        print(f"[graph] 实体 {self.G.number_of_nodes()} 个 / 三元组 {len(PRE_TRIPLES)} 条 / 社区 {len(comms)} 个")
        self.summ_M = emb.encode([s for s, _ in COMMUNITY_SUMMARIES])

    def _entities(self, query, topn=2):
        ents = sorted([n for n in self.G.nodes if n in query], key=len, reverse=True)
        return ents[:topn]

    def retrieve(self, query, k=5, scope="local"):
        return self._local(query, k) if scope == "local" else self._global(query, k)

    def _local(self, query, k):
        import numpy as np
        ents = self._entities(query)
        seen, scored = set(), []
        frontier = ents[:]
        for _ in range(3):  # 最多沿边走三跳
            nxt = []
            for e in frontier:
                edges = list(self.G.out_edges(e, data=True)) + list(self.G.in_edges(e, data=True))
                for h, t, d in edges:
                    key = (h, d["rel"], t)
                    if key not in seen:
                        seen.add(key)
                        scored.append((d["src"], f"{h} -{d['rel']}-> {t}"))
                        other = t if h == e else h
                        if other not in frontier:
                            nxt.append(other)
            nxt.sort(key=lambda x: 0 if self.G.out_degree(x) > 0 else 1)  # 实体优先展开
            frontier = nxt[:6]
        if not scored:  # 未命中实体 → 退化为全局查询
            return self._global(query, k)
        qv = self.emb.encode([query])[0]
        sims = self.emb.encode([t for _, t in scored]) @ qv
        best = {}
        for (doc, txt), s in zip(scored, sims):
            if doc not in best or s > best[doc]:
                best[doc] = float(s)
        ranked = sorted(best.items(), key=lambda kv: -kv[1])
        return _dedup(ranked, k)

    def _global(self, query, k):
        qv = self.emb.encode([query])[0]
        scores = self.summ_M @ qv
        top = scores.argsort()[::-1][:2]  # 取最相关的2个社区
        docs = []
        for ci in top:
            for d in COMMUNITY_SUMMARIES[ci][1]:
                if d not in docs:
                    docs.append(d)
        return docs[:k]

# ==============================================
# 7. WikiRAG（教材8.8节）：条目化知识库，检索单位是「条目」
#    条目 = 标题 + 聚合正文（重写而非复制原文），映射回来源文档评测
# ==============================================
PRE_ENTRIES = [
    ("云山大学", "云山大学位于岭南省云州市，建于1958年，省属重点大学，本科生约2.1万人，校训「格物致知」。地铁3号线「云大站」距东门200米，校巴5条线路。校历：春季学期3月2日开学。",
     ["doc01", "doc12", "doc14"]),
    ("人工智能学院", "人工智能学院成立于2018年，院长王启明教授，下设机器学习系、智能科学系、认知计算系。认知计算全国重点实验室依托本院建设。与新加坡南洋理工大学有本科联合培养项目（每年15人）。",
     ["doc02", "doc10", "doc15"]),
    ("计算机学院", "计算机学院前身是1985年成立的计算机系，院长陈国峰教授，设计算机科学与技术、软件工程两个专业。参与岭南超算中心运行管理。",
     ["doc03", "doc10"]),
    ("人工智能学院教师", "李文瀚：教授，2019年入职，研究检索增强生成，主讲《大模型通识课》，指导AI协会。赵婉晴：副教授，研究知识图谱，主讲《知识图谱导论》。孙浩然：讲师，研究强化学习，指导机器人战队。",
     ["doc04", "doc11"]),
    ("计算机学院教师", "周天宇：教授，研究分布式系统。林小雨：副教授，研究数据库系统。",
     ["doc05"]),
    ("《大模型通识课》", "《大模型通识课》（AI2101）：3学分32学时，春季学期开设，李文瀚主讲，面向全校本科生，无先修要求，教材《人工智能—深度学习大模型智能体》。",
     ["doc06"]),
    ("《知识图谱导论》", "《知识图谱导论》（AI3305）：2学分，秋季学期，赵婉晴主讲，面向人工智能学院研究生，需先修《数据结构》。",
     ["doc07", "doc08"]),
    ("奖学金体系", "国家奖学金8000元/年（约2%）；校长奖学金20000元/年（全校10人）；云山一等奖学金3000元/年（约5%）。",
     ["doc09"]),
    ("学生社团与校园生活", "AI协会：指导教师李文瀚，每周三晚论文研读。机器人战队：指导教师孙浩然，参加全国机器人大赛。图书馆藏书380万册，每天7:00-22:00开放，人工智能分馆在理科楼B座。",
     ["doc11", "doc13"]),
]

class WikiRAG:
    name = "WikiRAG"

    def __init__(self, emb):
        self.emb = emb
        self.M = emb.encode([b for _, b, _ in PRE_ENTRIES])

    def retrieve(self, query, k=5, scope=None):
        qv = self.emb.encode([query])[0]
        scores = self.M @ qv
        order = scores.argsort()[::-1][:3]  # 条目级Top-3
        docs = []
        for i in order:
            for d in PRE_ENTRIES[i][2]:
                if d not in docs:
                    docs.append(d)
        return docs[:k]

# ==============================================
# 8. 评测（教材式8-9 / 8-10）：检索单位（块/邻域三元组/条目）统一映射回文档
# ==============================================
def evaluate(systems, k=5, subset="all", detail=False):
    def qs_of(tp):
        return [q for q in QUESTIONS if q["type"] == tp and
                (subset == "all" or int(q["id"][1:]) <= 15)]

    print(f"\n==== 检索评测（文档级，Top-{k}） =====")
    print(f"{'格式':<12}{'问题类型':<10}{'Recall@' + str(k):>9}{'MRR':>8} 完整命中")
    for name, s in systems.items():
        for tp in ("fact", "multi", "global"):
            qs = qs_of(tp)
            recs, rrs, full = [], [], 0
            for q in qs:
                docs = s.retrieve(q["q"], k)
                ev = set(q["evidence"])
                got = ev & set(docs[:k])
                recs.append(len(got) / len(ev))
                rank = next((i + 1 for i, d in enumerate(docs) if d in ev), 0)
                rrs.append(1.0 / rank if rank else 0.0)
                full += (got == ev)
                if detail:
                    hit = "✔" if got else "✘"
                    print(f"  [{name}] {q['id']}({tp}) {hit} 检索={docs[:k]} 证据={q['evidence']} rank={rank}")
            print(f"{name:<14}{tp:<12}{sum(recs)/len(recs):>7.3f}{sum(rrs)/len(rrs):>8.3f} {full}/{len(qs)}")
    print("\n提示：把本表抄进报告表3；另抽3题手算MRR，用 --mrr_check 复核（式8-10）。")

# ==============================================
# 9. LLM生成环节（可选）：任一OpenAI兼容接口
#    环境变量：LLM_API_BASE / LLM_API_KEY / LLM_MODEL
# ==============================================
EXTRACT_PROMPT = """从下面这篇大学介绍文档中抽取「头实体-关系-尾实体」三元组，只输出JSON数组，
不要输出其他文字，格式如 [["李文瀚","任职于","人工智能学院"]]。
关系用2~4字短语（任职于/研究方向/主讲/成立于等），一篇文档抽5~10条。
文档：
{doc}"""

ANSWER_PROMPT = """你是问答助手。只依据下面资料回答问题：资料不足以回答时明确说「资料中未提及」，禁止编造。
问题：{q}
资料：
{ctx}
用1~3句话回答，句末用括号注明用到的资料编号。"""

def llm_chat(prompt, temperature=0.0):
    from openai import OpenAI
    base = os.environ.get("LLM_API_BASE")
    key = os.environ.get("LLM_API_KEY")
    model = os.environ.get("LLM_MODEL")
    if not (base and key and model):
        sys.exit("未配置LLM环境变量：请设置环境变量 LLM_API_BASE / LLM_API_KEY / LLM_MODEL")
    client = OpenAI(base_url=base, api_key=key)
    resp = client.chat.completions.create(model=model, temperature=temperature,
                                          messages=[{"role": "user", "content": prompt}])
    return resp.choices[0].message.content

def extract_triples():
    t0, out, bad = time.time(), [], 0
    for doc_id, text in CORPUS.items():
        raw = llm_chat(EXTRACT_PROMPT.replace("{doc}", text))
        try:
            arr = json.loads(raw[raw.index("["): raw.index("]") + 1])
        except Exception:
            bad += 1
            print(f"[warn] {doc_id} 解析失败：{raw[:60]}...")
            continue
        out += [(h, r, t, doc_id) for h, r, t in arr]
        print(f"{doc_id}: 抽到 {len(arr)} 条")
    print(f"\n共 {len(out)} 条 / 解析失败 {bad} 篇，耗时 {time.time() - t0:.0f}s。")
    print("请人工抽检后，把高质量三元组替换进脚本 PRE_TRIPLES。")

# ==============================================
# 10. 主入口
# ==============================================
def main():
    ap = argparse.ArgumentParser(description="实验作业四：三种RAG对比实验")
    ap.add_argument("--mode", choices=["eval", "vector", "graph", "wiki", "extract"], default="eval")
    ap.add_argument("--q", default="Q01", help="问题编号，如 Q01")
    ap.add_argument("--scope", choices=["local", "global"], default="local")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--emb", default="paraphrase-multilingual-MiniLM-L12-v2")
    ap.add_argument("--subset", choices=["all", "builtin"], default="all", help="评测问题集：all=20题，builtin=内置15题")
    ap.add_argument("--detail", action="store_true", help="逐题打印检索结果")
    ap.add_argument("--mrr_check", default=None, help="如3,1,2: 打印MRR手算过程")
    args = ap.parse_args()

    if args.mrr_check:
        # 手算复核: MRR = (1/3+1/1+1/2)/3 = 0.611
        ranks = [int(x) for x in args.mrr_check.split(",")]
        m = sum(1.0 / r for r in ranks) / len(ranks)
        print(f"MRR = (" + " + ".join(f"1/{r}" for r in ranks) + f") / {len(ranks)} = {m:.3f}")
        return

    if args.mode == "extract":
        extract_triples()
        return

    emb = Embedder(args.emb)
    systems = {}
    if args.mode in ("eval", "vector"):
        systems["Vector RAG"] = VectorRAG(emb)
    if args.mode in ("eval", "graph"):
        systems["GraphRAG"] = GraphRAG(emb)
    if args.mode in ("eval", "wiki"):
        systems["WikiRAG"] = WikiRAG(emb)

    if args.mode == "eval":
        evaluate(systems, args.k, args.subset, args.detail)
        return

    name = {"vector": "Vector RAG", "graph": "GraphRAG", "wiki": "WikiRAG"}[args.mode]
    q = next(q for q in QUESTIONS if q["id"].upper() == args.q.upper())
    docs = systems[name].retrieve(q["q"], args.k, args.scope)
    print(f"\n[{q['id']}·{q['type']}]: {q['q']}")
    print("检索到的文档: ", docs)
    ctx = "\n".join(f"[{d}] {CORPUS[d]}" for d in docs)
    try:
        print("\n回答: ", llm_chat(ANSWER_PROMPT.replace("{q}", q["q"]).replace("{ctx}", ctx)))
    except SystemExit:
        print("\n（未配置LLM API，仅输出检索结果与上下文）\n" + ctx)

if __name__ == "__main__":
    main()
