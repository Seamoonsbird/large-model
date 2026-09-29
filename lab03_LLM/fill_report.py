# -*- coding: utf-8 -*-
import os
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

os.chdir(os.path.dirname(os.path.abspath(__file__)))
DOC = "学号_姓名_实验作业三.docx"
doc = Document(DOC)

def set_text(p, text):
    """清空段落并写入文本，保留首个 run 的字体。"""
    for r in list(p.runs):
        r._element.getparent().remove(r._element)
    run = p.add_run(text)
    return run

def set_cell(cell, text, bold=False, size=10):
    cell.text = ""
    p = cell.paragraphs[0]
    run = p.add_run(text)
    run.font.size = Pt(size)
    run.bold = bold

def add_img_to_cell(cell, img_path, width=5.5):
    cell.text = ""
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.add_picture(img_path, width=Inches(width))

# ============ 封面 ============
for p in doc.paragraphs:
    if p.text.startswith("完成日期"):
        set_text(p, "完成日期：2026 年 9 月 29 日")

# ============ 1.1 实验目的 ============
purpose = ("本实验以约 2000 字的唐诗为语料，用 PyTorch 亲手实现一个约 50 万参数的最小 GPT，"
           "把教材第 5 章的 Transformer 公式变成可运行代码。具体包括：(1) 理解“预测下一字符”"
           "的自监督目标，掌握字符级分词、上下文窗口截取与标签错开一位；(2) 搭建词嵌入+可学习位置编码+"
           "Pre-Norm Transformer 块+权重共享的最小 GPT；(3) 在 CPU 上完成基线训练、绘制 loss 曲线、"
           "自回归生成诗句并观察温度/top-k 的影响；(4) 通过学习率、层数、嵌入维数、上下文长度、位置编码"
           "五组控制变量实验理解各超参数作用，并手算参数量与代码打印值相互印证。")
# 定位 purpose 提示段，在其后连续的下划线段中填入目的，其余清空
ps = doc.paragraphs
for i, p in enumerate(ps):
    if p.text.strip().startswith("提示：参考指南第一节"):
        set_text(p, purpose)
        j = i + 1
        while j < len(ps) and ps[j].text.strip().startswith("___"):
            set_text(ps[j], ""); j += 1
        break

# ============ 表1 实验环境 ============
t0 = doc.tables[0]
set_cell(t0.rows[1].cells[1], "Linux（x86_64，2 vCPU）")
set_cell(t0.rows[2].cells[1], "Python 3.12.11")
set_cell(t0.rows[3].cells[1], "torch 2.14.0+cpu")
set_cell(t0.rows[4].cells[1], "matplotlib 3.11.0")
set_cell(t0.rows[5].cells[1], "TRAE IDE（本报告在命令行复现训练）")
set_cell(t0.rows[5].cells[2], "使用功能：AI 对话生成代码骨架、逐行代码解释、形状类报错排查、调参方案检查")

# ============ 2.2 基线结果 ============
for p in doc.paragraphs:
    if p.text.startswith("参数量："):
        set_text(p, "参数量：502,656（约 50.3 万）　　最终 loss：0.0207　　"
                    "初始 loss（≈ln 699≈6.55）：6.5638　　耗时：4.0 min（2000 步，约 8.2 it/s）")

# 基线 loss 曲线（表2）
add_img_to_cell(doc.tables[2].rows[0].cells[0], "loss_curve_L2_E128_lr0.001.png")

# 基线生成示例（表3）
with open("generated_base.txt", encoding="utf-8") as f:
    base_gen = f.read().strip()
set_cell(doc.tables[3].rows[0].cells[0], base_gen, size=9)

# ============ 3 参数量手算（表4） ============
t4 = doc.tables[4]
set_cell(t4.rows[1].cells[2], "89,472")
set_cell(t4.rows[2].cells[2], "16,384")
set_cell(t4.rows[3].cells[2], "393,216")
set_cell(t4.rows[4].cells[2], "499,072（手算）；502,656（代码打印）")

# ============ 4 对照实验（表5） ============
t5 = doc.tables[5]
rows_data = [
    # final loss, quality, phenomenon
    ("1e-2: 0.1663；1e-4: 0.6209",
     "1e-2：★★；1e-4：★",
     "1e-2 前期剧烈震荡（前100步均值5.76，基线仅3.64），学习率过大训练不稳，出现“雁雁雁”“异乡为异乡为异乡”式重复；"
     "1e-4 收敛过慢，1000步远未拟合，生成是无意义字词拼凑。三组初始 loss 都从≈6.55出发。"),
    ("1 层: 0.0294；4 层: 0.0275",
     "1 层：★★★★；4 层：★★★★",
     "单层即可很好地背诵小语料，loss 已接近基线；4 层 loss 仅微降，小语料下加深收益有限，"
     "但耗时从1.5min增至4.2min。说明在 2000 字语料上模型容量已过剩，瓶颈在数据而非网络深度。"),
    ("64(头2): 0.0426；256(头8): 0.0304",
     "64：★★★；256：★★★★",
     "64 维参数仅15.3万，loss略高、偶有漏字错字（如“芳草萋鹦鹉洲”漏“萋”）；"
     "256 维参数达179万（约3.6倍）、耗时5.9min，loss 未明显更优，生成质量提升有限——小语料下大嵌入维属冗余。"),
    ("T=32: 0.1157",
     "★★★",
     "上下文缩短到32字，模型每步只能看到更短片段，连贯性下降，出现叠字（“密密缝”“觉觉浅”）；"
     "但因语料短诗句多为4句内，仍能生成大体成句的片段。"),
    ("无位置: 0.0392",
     "★★★",
     "去掉位置编码后 loss 仍可降到0.04（字符共现仍可学），但顺序信息丢失，生成出现重复堆叠"
     "（“悠悠悠”“一片孤城万仞山”反复出现），不再能区分字的先后次序。"),
    ("T=0.5: 0.0207；T=1.5: 0.0207（同一基线模型）",
     "0.5：★★★★★；1.5：★★",
     "温度只影响采样不影响权重：0.5 近乎逐字背诵最可能诗句、确定性高但多样性低；"
     "1.5 概率被拉平，出现拼接错乱（“春风不是玉关”“到乡翻似烂柯千帆过”），胡言增多。"),
]
for i, (loss, qual, phen) in enumerate(rows_data, start=1):
    set_cell(t5.rows[i].cells[4], loss, size=9)
    set_cell(t5.rows[i].cells[5], qual, size=9)
    set_cell(t5.rows[i].cells[6], phen, size=9)

# ============ 对照实验曲线 ============
add_img_to_cell(doc.tables[6].rows[0].cells[0], "loss_curve_L2_E128_lr0.01.png", width=5.8)
add_img_to_cell(doc.tables[7].rows[0].cells[0], "loss_curve_L1_E128_lr0.001.png", width=5.8)
add_img_to_cell(doc.tables[8].rows[0].cells[0], "loss_curve_L2_E64_lr0.001.png", width=5.8)
add_img_to_cell(doc.tables[9].rows[0].cells[0], "loss_curve_L2_E128_lr0.001_T32.png", width=5.8)
add_img_to_cell(doc.tables[10].rows[0].cells[0], "loss_curve_no_pos.png", width=5.8)

# ============ 4.x 分析文字 ============
analyses = [
    "1e-2 并未发散成 NaN，但前期 loss 在高位剧烈震荡、最终停在0.17，明显差于基线0.02；"
    "1e-4 则收敛太慢，1000步时 loss 仍有0.62。三者初始 loss 都≈6.55，说明初始化方差一致，差别完全来自学习率。",
    "1 层与 4 层最终 loss（0.029 vs 0.028）几乎打平，说明在 2000 字小语料上，2 层已足够拟合；"
    "4 层参数量翻倍、耗时近3倍却无收益，印证了“小数据下加深网络容易过拟合/无收益”的判断。",
    "64 维参数15万、256维参数179万（约3.6倍），训练耗时分别约1.1min与5.9min；"
    "但最终 loss（0.043 vs 0.030）差距远小于参数量差距，生成质量提升也不明显，容量与训练成本并不线性转化为质量。",
    "T=32 时模型“视野”短，无法利用跨句的押韵与对仗，生成连贯性下降、叠字增多；"
    "说明上下文长度决定了模型能建模多长的依赖关系。",
    "无位置编码时 loss 仍能下降（自注意力靠字符共现即可预测），但生成的诗出现重复堆叠、语序混乱，"
    "因为自注意力对位置“排列不变”，模型不知道字的先后——与思考题(1)呼应。",
]
# 遇到 4.x 小节标题后，下一个下划线段落填入对应分析，其后连续下划线清空；其它区域不动
mode = 0   # 0=正常, 1=待填, 2=清空后续连续下划线
ai = -1
for p in doc.paragraphs:
    t = p.text.strip()
    if t[:3] in ("4.1", "4.2", "4.3", "4.4", "4.5") and "实验" in t:
        ai = ["4.1","4.2","4.3","4.4","4.5"].index(t[:3]); mode = 1
    elif t.startswith("___"):
        if mode == 1:
            set_text(p, analyses[ai]); mode = 2
        elif mode == 2:
            set_text(p, "")
    else:
        if mode == 2:
            mode = 0

# ============ 5.1 采样记录表（表11） ============
t11 = doc.tables[11]
# 读采样结果
sampling = {}
with open("sampling_results.txt", encoding="utf-8") as f:
    blocks = f.read().split("=====")
for b in blocks:
    b = b.strip()
    if not b: continue
    head = b.split("=====")[0].strip() if "=====" in b else ""
# 手动摘录每组代表性原文（取自 sampling_results.txt）
s_excerpt = {
    1: ("春。半亩方塘一鉴开，天光云影共徘徊……人生自古谁无死，留取丹心照汗青。\n"
        "月出惊山鸟，时鸣春涧中。红豆生南国，春来发几枝……"),
    2: ("春。半亩方塘一鉴开，天光云影共徘徊……（与组1几乎逐字相同，更确定）\n"
        "月，对影成三人。小时不识月，呼作白玉盘……"),
    3: ("春。巴山楚水凄凉地……到乡翻似烂柯千帆过，病树前头万木春。\n"
        "春风不是玉关。……路无自古今日曛，路（明显错乱、字词拼接）"),
    4: ("春。半亩方塘一鉴开……（与组1高度重合，top5已覆盖背诵路径）\n"
        "月，呼作白玉盘。又疑瑶台镜，飞在青云端……"),
    5: ("春。半亩方塘一鉴开……（与组1/组4几乎一致）\n"
        "月声，花落知多少。白日依山尽，黄河入海流……"),
}
s_note = {
    1: "基线：既能成句背诵，又有一定变化，效果最均衡。",
    2: "温度低→分布尖锐→近乎贪心，确定性最高、重复原诗，多样性最低。",
    3: "温度高→分布平坦→采样到低概率字，诗句开始错乱、拼接，胡言增多。",
    4: "top_k=5→只保留最高概率5个字，因模型已背熟，结果与基线几乎一致，更保守。",
    5: "不限 top_k→理论上可采样任意字，但因分布极尖锐，实际仍集中在高概率区，与基线差异不大。",
}
for r in range(1, 6):
    set_cell(t11.rows[r].cells[3], s_excerpt[r], size=9)
    set_cell(t11.rows[r].cells[4], s_note[r], size=9)

# 表12 采样对比（放文字摘要）
set_cell(doc.tables[12].rows[0].cells[0],
         "完整采样原文见 sampling_results.txt。\n"
         "组1(T=1.0,k=20)与组4(k=5)、组5(k=699)生成几乎一致：因小语料模型已背熟、概率分布极尖锐，"
         "top_k 大小影响被温度主导；组2(T=0.5)更确定地逐字背诵；组3(T=1.5)出现“春风不是玉关”“烂柯千帆过”等错乱拼接。",
         size=9)

# ============ 5.2 分析 ============
a52 = ("从“确定性↔多样性”看：温度越低，softmax 分布越尖锐，采样越接近贪心取最高概率字，生成越确定、越像原诗，"
       "但几乎没有新意；温度越高，分布越平，采样越接近均匀，多样性上去了却开始出现胡言。"
       "从“重复↔胡言”看：T=0.5 与 top_k=5 会加剧重复（几乎照抄同一句诗），T=1.5 则打破重复但引入错乱拼接。"
       "本组最优组合为 temperature=1.0、top_k=20：既能基本成句、保留诗的格式与常见用字，又有一定变化不致死板；"
       "在本实验已背熟的小语料上，top_k=5 也可接受（更稳），而 top_k 放开到全词表并无额外收益。")
for i, p in enumerate(doc.paragraphs):
    if p.text.strip().startswith("提示：从“确定性"):
        set_text(p, a52)
        j = i + 1
        while j < len(doc.paragraphs) and doc.paragraphs[j].text.strip().startswith("___"):
            set_text(doc.paragraphs[j], ""); j += 1
        break

# ============ 6 AI 协作记录（表13） ============
t13 = doc.tables[13]
ai_rows = [
    ("让 AI 按“数据→分词器→模型→训练→采样”顺序逐段生成 min_llm.py 骨架",
     "“请按数据流顺序生成字符级 GPT：先语料与 CharTokenizer，再 MiniGPT，再训练循环与 generate”",
     "采纳骨架但逐段自己读懂后再拼；修正了 PDF 参考代码中的 OCR 笔误（n_embd、torch.topk、generate 括号与缩进）。"),
    ("让 AI 解释 CausalSelfAttention 各步张量形状",
     "“选中 CausalSelfAttention，解释 QKV 一次投影后 view+transpose 如何拆成多头，以及因果掩码为何填 -inf”",
     "采纳解释：(B,T,C)→(B,head,T,head_dim)；自己在纸上核对形状后确认 masked_fill 条件应为 mask==0。"),
    ("排查 argparse 报错 unrecognized arguments: --n_embd",
     "运行 exp3 时报 unrecognized arguments，自己先读 main() 发现参数名是 --n_embed",
     "未让 AI 改，自行把脚本中 --n_embd 改为 --n_embed 后重跑成功。"),
]
for i, (scene, prompt, out) in enumerate(ai_rows, start=1):
    set_cell(t13.rows[i].cells[1], scene, size=9)
    set_cell(t13.rows[i].cells[2], prompt, size=9)
    set_cell(t13.rows[i].cells[3], out, size=9)

# ============ 7 思考题 ============
think = [
    # (1)
    "去掉位置编码后，模型仍能学到“哪些字符常一起出现”（字符共现统计），所以 loss 仍可降到0.04、生成大体仍是五言/七言句式；"
    "但自注意力本身对输入顺序是“排列不变”的——打乱字序不改变注意力权重之和，模型不知道第1个字和第5个字谁先谁后。"
    "于是生成出现“悠悠悠”“一片孤城万仞山”反复堆叠、语序混乱等现象，诗虽有常见字词却不再成句。"
    "这说明位置编码的作用正是向模型注入序列的先后次序信息。",
    # (2)
    "手算：V×C=699×256=178,944；T×C=128×256=32,768；L×12C²=3×12×256²=2,359,296；合计≈2,571,008（约257万）。"
    "基线手算为499,072，故增长约 2,571,008/499,072≈5.15 倍。实测上 exp3 把 n_embd 从128加到256，"
    "1000步耗时由约3min增至5.9min；exp2 层数从2加到4，耗时约翻倍。可见参数量随 C²、近似随 L 线性增长，"
    "训练成本也随之上升，但在仅2000字的小语料上，容量增大并未带来 loss 或生成质量的明显提升——"
    "容量必须与数据规模匹配，盲目堆大只会增加训练成本。",
    # (3)
    "T→0 时，logits/T 中最大值被无限放大，softmax 趋近 one-hot，采样退化为“贪心 argmax”（永远取概率最高的字），"
    "输出确定、可复现但高度重复；T→∞ 时，logits/T→0，softmax 趋近均匀分布，采样退化为“从词表随机乱取”，"
    "输出完全随机、胡言乱语。实践中：摘要、翻译、纠错、代码补全等要求稳定准确的任务用低温（0.2~0.5）；"
    "创意写作、头脑风暴、对话生成等需要多样有趣的场景用中高温（0.8~1.2）。本组 T=0.5 近乎逐字背诵，"
    "T=1.5 则出现大量错乱拼接，正印证了这两个极端。",
    # (4)
    "字符级分词优点：词表极小（本实验仅699）、零预处理、永远不会有未登录字、实现简单；"
    "缺点：序列被拉得很长（一个汉字就是一个 token，单 token 信息量少），模型要在更长序列上学依赖，效率低。"
    "BPE（字节对编码）在“词表大小”与“序列长度”之间折中：它把高频的常用字/词组合并成一个 token，"
    "既把词表控制在3万~15万，又把序列长度压到远短于字符级。真实 LLM 都用 BPE 类分词，是因为"
    "字符级在海量多语言语料上序列过长、算力开销大；而整词分词又会让词表爆炸、未登录词难处理，BPE 是两者的平衡点。",
    # (5)
    "三方面比较：数据量上，本实验仅约2000字、72首诗，真实 LLM 预训练用数万亿 token 的多语料；"
    "参数量上，本实验约50万，真实 LLM 为数十亿~上万亿；训练目标上，本实验只有下一字符预测（交叉熵），"
    "真实 LLM 还要经过指令微调与 RLHF 来对齐人类意图。把本实验语料扩大1000倍（约200万字），"
    "依然不能得到“会聊天”的模型：一是200万字相对预训练量级仍是沧海一粟，泛化远远不够；"
    "二是只有下一 token 预测只能学到语言的统计接续，缺乏“听指令、按对话格式回应”的监督信号；"
    "三是缺少 RLHF/人类反馈，模型不会主动、礼貌地与人对话。它最多成为一个“能生成更多古诗文风格文本”的模型。",
    # (6)
    "AI 最有价值的环节：(1) 按数据流快速生成整体骨架，省去查 API 的时间；(2) 逐行解释张量形状与注意力公式，"
    "帮助快速理解 view/transpose 的维度变化；(3) 对照实验方案的单一变量检查。"
    "必须自己读懂才能改对的地方：因果掩码为何对 mask==0 的位置填 -inf（而非0），否则会泄露未来信息；"
    "多头拆分时 view 后必须 transpose 才能把 head 维提前、且最后要 contiguous 再 view 回 (B,T,C)，顺序错了形状就崩；"
    "权重共享 self.head.weight = self.tok_emb.weight 必须在初始化之前/之后正确赋值，否则输出层与词嵌入不共享。"
    "这些地方 AI 可以提示，但形状与语义必须自己在纸上推一遍才能真正改对。",
]

# 找到思考题后的下划线段
q_starts = ["(1) 结合 exp5", "(2) 手算：若 n_embd", "(3) 温度 T→0", "(4) 字符级分词",
            "(5) 本实验是", "(6) 回顾你与 TRAE"]
para_texts = [p.text for p in doc.paragraphs]
# 收集每题后面连续的下划线段落，把答案放第一段，其余清空
qi = 0
i = 0
while i < len(doc.paragraphs):
    txt = doc.paragraphs[i].text.strip()
    if any(txt.startswith(q) for q in q_starts):
        # 后续连续下划线段
        j = i + 1
        filled = False
        while j < len(doc.paragraphs) and doc.paragraphs[j].text.strip().startswith("___"):
            if not filled:
                set_text(doc.paragraphs[j], think[qi]); filled = True
            else:
                set_text(doc.paragraphs[j], "")
            j += 1
        qi += 1
        i = j
    else:
        i += 1

doc.save(DOC)
print("saved", DOC)
