# -*- coding: utf-8 -*-
"""
自制语言模型的知识蒸馏 —— 实验作业八 (纯 CPU 可运行)

单文件、零外部数据:
  * 内置「仿古诗句」合成语料生成器(可复现, 不依赖网络与版权)
  * 字符级 GPT(教师/学生同架构、不同宽度)
  * 三种训练: 教师监督训练、学生从零训练(hard label)、学生蒸馏训练(soft label)
  * 温度 T / 权重 alpha 消融、中间层特征蒸馏、自蒸馏、动态量化导出

常用命令:
  python distill_lm.py --mode quick     # 30 秒自检
  python distill_lm.py --mode teacher --retrain
  python distill_lm.py --mode all
  python distill_lm.py --mode ablate
  python distill_lm.py --mode feat
  python distill_lm.py --mode self
  python distill_lm.py --mode quant
"""
import argparse
import json
import math
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(max(1, min(8, os.cpu_count() or 4)))

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs_distill")

# =======================================================================
# 1. 语料: 可复现的合成「仿古诗句」
# 四个槽位 + 两个收尾模板, 使下一个字的分布具有真实的多样性与结构,
# 这正是教师 soft label 能传给学生的「暗知识」。
# =======================================================================
SUBJ = ["小明", "小红", "阿远", "书生", "渔夫", "牧童", "旅人", "诗人",
        "少年", "故人", "归客", "山僧", "孤客", "游子"]
PLACE = ["江边", "山寺", "长亭", "渡口", "竹林", "荒村", "城楼", "野渡",
         "溪桥", "松间", "篱下", "窗前", "陌上", "舟中"]
VERB = ["望见", "听见", "想起", "送去", "寻到", "梦见", "独对", "遥看",
        "静听", "又逢", "恰遇", "空忆"]
OBJ = ["明月", "落花", "孤舟", "归雁", "残灯", "细雨", "青山", "晚钟",
       "白雪", "寒江", "疏星", "暮云", "秋草", "远帆"]
MOOD = ["心中惆怅", "不觉天明", "欣然归去", "泪湿衣襟", "长叹一声",
        "默然无语", "忽忆旧游", "欲语还休", "此意难平", "一笑而别"]

POEMS = [
    "床前明月光, 疑是地上霜。举头望明月, 低头思故乡。",
    "春眠不觉晓, 处处闻啼鸟。夜来风雨声, 花落知多少。",
    "白日依山尽, 黄河入海流。欲穷千里目, 更上一层楼。",
    "千山鸟飞绝, 万径人踪灭。孤舟蓑笠翁, 独钓寒江雪。",
    "空山不见人, 但闻人语响。返景入深林, 复照青苔上。",
    "红豆生南国, 春来发几枝。愿君多采撷, 此物最相思。",
    "独坐幽篁里, 弹琴复长啸。深林人不知, 明月来相照。",
    "月落乌啼霜满天, 江枫渔火对愁眠。姑苏城外寒山寺, 夜半钟声到客船。",
    "两个黄鹂鸣翠柳, 一行白鹭上青天。窗含西岭千秋雪, 门泊东吴万里船。",
    "朝辞白帝彩云间, 千里江陵一日还。两岸猿声啼不住, 轻舟已过万重山。",
    "日照香炉生紫烟, 遥看瀑布挂前川。飞流直下三千尺, 疑是银河落九天。",
    "渭城朝雨浥轻尘, 客舍青青柳色新。劝君更尽一杯酒, 西出阳关无故人。",
    "寒雨连江夜入吴, 平明送客楚山孤。洛阳亲友如相问, 一片冰心在玉壶。",
    "独在异乡为异客, 每逢佳节倍思亲。遥知兄弟登高处, 遍插茱萸少一人。",
    "清明时节雨纷纷, 路上行人欲断魂。借问酒家何处有, 牧童遥指杏花村。",
    "朱雀桥边野草花, 乌衣巷口夕阳斜。旧时王谢堂前燕, 飞入寻常百姓家。",
]


def build_corpus(n_lines=3000, seed=0):
    """用固定随机种子生成训练语料; 留出集换一个种子, 检验泛化而非记忆。"""
    rng = random.Random(seed)
    out = []
    for i in range(n_lines):
        s, p, v, o, m = (rng.choice(x) for x in (SUBJ, PLACE, VERB, OBJ, MOOD))
        if i % 3 == 0:
            out.append("%s%s%s%s, %s。" % (s, p, v, o, m))
        elif i % 3 == 1:
            out.append("%s在%s了一夜%s, %s。" % (s, p, o, m))
        else:
            out.append("%s%s道:「%s%s。」%s。" % (s, v, o, rng.choice(["正好", "可怜", "可惜", "依旧"]), m))
    out.extend(POEMS)          # 真实唐诗, 只放一份, 提供与合成句不同的长句结构
    return "\n".join(out)


# =============================================================================
# 2. 字符级分词
# =============================================================================
class Tokenizer:
    def __init__(self, text):
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = chars
        self.vocab_size = len(chars)

    def encode(self, s):
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)


# =============================================================================
# 3. 最小 GPT (Pre-Norm, 与作业三同构, 只改宽度/深度)
# =============================================================================
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        C = cfg["n_embd"]
        assert C % cfg["n_head"] == 0
        self.n_head = cfg["n_head"]
        self.c_attn = nn.Linear(C, 3 * C)
        self.c_proj = nn.Linear(C, C)
        self.drop = nn.Dropout(cfg["dropout"])
        self.register_buffer(
            "mask", torch.tril(torch.ones(cfg["block_size"], cfg["block_size"]))
            .view(1, 1, cfg["block_size"], cfg["block_size"]))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = self.drop(F.softmax(att, dim=-1))
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        C = cfg["n_embd"]
        self.fc = nn.Linear(C, 4 * C)
        self.proj = nn.Linear(4 * C, C)
        self.drop = nn.Dropout(cfg["dropout"])

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg["n_embd"])
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg["n_embd"])
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, vocab_size, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(vocab_size, cfg["n_embd"])
        self.pos_emb = nn.Embedding(cfg["block_size"], cfg["n_embd"])
        self.drop = nn.Dropout(cfg["dropout"])
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg["n_layer"])])
        self.ln_f = nn.LayerNorm(cfg["n_embd"])
        self.head = nn.Linear(cfg["n_embd"], vocab_size, bias=False)
        self.apply(self._init)
        self.n_param = sum(p.numel() for p in self.parameters())

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None, need_hidden=False):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        if need_hidden:
            return logits, loss, x
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=60, temperature=0.8, top_k=40):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg["block_size"]:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        self.train()
        return idx


# ========================================================================
# 4. 批采样: scratch 与 kd 使用同一 generator 种子, 保证数据顺序一致
# ========================================================================
def get_batch(data, bs, block, gen):
    ix = torch.randint(len(data) - block - 1, (bs,), generator=gen)
    x = torch.stack([data[i:i + block] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block] for i in ix])
    return x, y


@torch.no_grad()
def evaluate(model, data, bs, block, iters=20, seed=1234):
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    tot = 0.0
    for _ in range(iters):
        x, y = get_batch(data, bs, block, gen)
        _, loss = model(x, y)
        tot += loss.item()
    model.train()
    return tot / iters


@torch.no_grad()
def agreement(student, teacher, data, bs, block, iters=10, seed=4321):
    """学生与教师在留出集上的 top-1 一致率, 衡量「学到了多少」"""
    student.eval()
    teacher.eval()
    gen = torch.Generator().manual_seed(seed)
    hit = tot = 0
    for _ in range(iters):
        x, _ = get_batch(data, bs, block, gen)
        a = student(x)[0].argmax(-1)
        b = teacher(x)[0].argmax(-1)
        hit += (a == b).sum().item()
        tot += b.numel()
    student.train()
    teacher.train()
    return hit / tot


# =================================================================
# 5. 训练: hard label / soft label (KD)
# =================================================================
def train(model, data, iters, bs, block, lr=3e-3, seed=0,
          teacher=None, T=4.0, alpha=0.5, feat_weight=0.0,
          log_every=50, tag="", val=None):
    """
    alpha = 0  -> 纯交叉熵(从零训练的对照组)
    alpha > 0  -> L = (1-alpha) * CE + alpha * T^2 * KL(教师软标签 || 学生软标签)
    feat_weight -> 额外加最后一层隐状态 MSE (FitNet 式特征蒸馏)
    """
    proj = None
    if teacher is not None and feat_weight > 0:
        proj = nn.Linear(model.cfg["n_embd"], teacher.cfg["n_embd"])
    params = list(model.parameters()) + (list(proj.parameters()) if proj is not None else [])
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    gen = torch.Generator().manual_seed(seed + 1)
    hist = []
    ce_hist = []
    if teacher is not None:
        teacher.eval()
    model.train()
    t0 = time.time()
    for it in range(iters):
        x, y = get_batch(data, bs, block, gen)
        logits = model(x)[0]
        ce = F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1))
        loss = ce
        kd = None
        if teacher is not None and alpha > 0:
            with torch.no_grad():
                t_logits = teacher(x)[0]
            # 关键: 先 reshape 成 (B*T, V) 再算 KL。
            # 若直接对 (B,T,V) 调用 kl_div(reduction="batchmean"), 它只除以 B,
            # 序列维会被重复累加, KL 被放大 T 倍 (T=96 时数值大得离谱)。
            V = logits.size(-1)
            s_lp = F.log_softmax(logits.reshape(-1, V) / T, dim=-1)
            t_p = F.softmax(t_logits.reshape(-1, V) / T, dim=-1)
            kd = F.kl_div(s_lp, t_p, reduction="batchmean") * (T * T)
            loss = (1 - alpha) * ce + alpha * kd
        if proj is not None:
            _, _, hs = model(x, need_hidden=True)
            with torch.no_grad():
                _, _, ht = teacher(x, need_hidden=True)
            loss = loss + feat_weight * F.mse_loss(proj(hs), ht)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        hist.append(loss.item())
        ce_hist.append(ce.item())
        if (it + 1) % log_every == 0 or it == 0:
            extra = ""
            if kd is not None:
                extra = " ce %.4f kd(T^2) %.4f" % (ce.item(), kd.item())
            msg = "[%s] iter %4d/%d loss %.4f%s (%.1fs)" % (
                tag or "train", it + 1, iters, loss.item(), extra, time.time() - t0)
            if val is not None and (it + 1) % (log_every * 2) == 0:
                msg += " | val %.4f" % evaluate(model, val, bs, block)
            print(msg, flush=True)
    model._ce_hist = ce_hist
    return hist, time.time() - t0


# =============================================================================
# 6. 主流程
# =============================================================================
TEACHER_CFG = dict(n_layer=2, n_head=4, n_embd=160, block_size=96, dropout=0.0)
STUDENT_CFG = dict(n_layer=2, n_head=4, n_embd=64, block_size=96, dropout=0.0)


def make_data(block):
    train_text = build_corpus(3000, seed=0)
    val_text = build_corpus(300, seed=999)   # 同语法、不同种子 -> 留出集
    tok = Tokenizer(train_text)
    train_ids = torch.tensor(tok.encode(train_text), dtype=torch.long)
    val_ids = torch.tensor(tok.encode(val_text), dtype=torch.long)
    return tok, train_ids, val_ids


def get_teacher(tok, args, res, bs, block):
    path = os.path.join(OUT_DIR, "teacher.pt")
    cfg_key = "%d" % TEACHER_CFG["n_embd"]
    if os.path.exists(path) and not args.retrain:
        t = GPT(tok.vocab_size, TEACHER_CFG)
        t.load_state_dict(torch.load(path, map_location="cpu"))
        tv = evaluate(t, val_ids_global, bs, block)
        res["teacher"] = {"params": t.n_param, "val_loss": tv, "ppl": math.exp(tv),
                          "time_s": 0.0, "curve": [], "cached": True}
        print("[teacher] 复用 teacher.pt val_loss=%.4f ppl=%.2f" % (tv, math.exp(tv)), flush=True)
        return t
    torch.manual_seed(args.seed)
    t = GPT(tok.vocab_size, TEACHER_CFG)
    print("[teacher] params = %.3f M" % (t.n_param / 1e6), flush=True)
    th, ttime = train(t, train_ids_global, args.teacher_iters, bs, block,
                      lr=args.lr, seed=args.seed, log_every=args.log_every,
                      tag="teacher", val=val_ids_global)
    tv = evaluate(t, val_ids_global, bs, block)
    torch.save(t.state_dict(), path)
    res["teacher"] = {"params": t.n_param, "val_loss": tv, "ppl": math.exp(tv),
                      "time_s": ttime, "curve": th, "cached": False}
    print("[teacher] val_loss=%.4f ppl=%.2f %.1fs" % (tv, math.exp(tv), ttime), flush=True)
    return t


def gen_samples(models, tok, prompt="月", n=60):
    idx = torch.tensor([[tok.stoi.get(c, 0) for c in prompt]], dtype=torch.long)
    out = {}
    for name, m in models.items():
        ids = m.generate(idx.clone(), max_new_tokens=n, temperature=0.8, top_k=40)
        out[name] = tok.decode(ids[0].tolist())
    return out


def plot_curves(res, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("[plot] 跳过绘图 (%s)" % e)
        return
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for k, lab in (("teacher", "Teacher"), ("scratch", "Student-scratch"), ("kd", "Student-KD")):
        if k in res and res[k].get("curve"):
            c = res[k]["curve"]
            ax[0].plot(range(1, len(c) + 1), c, label=lab, linewidth=1.2)
    ax[0].set_xlabel("iteration")
    ax[0].set_ylabel("training loss")
    ax[0].set_title("training loss")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    if "ablation" in res:
        for T in sorted({r["T"] for r in res["ablation"]}):
            xs, ys = [], []
            for r in res["ablation"]:
                if r["T"] == T:
                    xs.append(r["alpha"])
                    ys.append(r["val_loss"])
            ax[1].plot(xs, ys, marker="o", label="T=%d" % T, linewidth=1.2)
        ax[1].set_xlabel("alpha")
        ax[1].set_ylabel("val loss")
        ax[1].set_title("temperature / weight ablation")
        ax[1].legend(fontsize=8)
        ax[1].grid(alpha=0.3)
    else:
        ax[1].bar(["scratch", "kd"],
                  [res["scratch"]["val_loss"], res["kd"]["val_loss"]])
        ax[1].set_title("val loss")
        ax[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print("[plot] saved %s" % path, flush=True)


def save(res, tag="run"):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "results_%s.json" % tag), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)


val_ids_global = None
train_ids_global = None


def main():
    global val_ids_global, train_ids_global, OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all",
                    choices=["quick", "teacher", "all", "ablate", "feat", "self", "quant", "sweep", "tscan", "tuned", "probe", "tscan_multi", "paired", "corpus_test", "scratch_multi", "gen"])
    ap.add_argument("--teacher-iters", type=int, default=200)
    ap.add_argument("--student-iters", type=int, default=120)
    ap.add_argument("--abl-iters", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--T", type=float, default=4.0)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--feat", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--retrain", action="store_true", help="忽略已缓存的 teacher.pt, 重新训练教师")
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()

    if args.mode == "quick":
        OUT_DIR = os.path.join(OUT_DIR, "quick")
        args.teacher_iters, args.student_iters, args.abl_iters = 20, 10, 8
        args.batch_size, args.log_every = 8, 5

    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(args.seed)
    tok, train_ids, val_ids = make_data(TEACHER_CFG["block_size"])
    train_ids_global, val_ids_global = train_ids, val_ids
    bs, block = args.batch_size, TEACHER_CFG["block_size"]
    print("[data] vocab=%d train_chars=%d val_chars=%d block=%d batch=%d"
          % (tok.vocab_size, len(train_ids), len(val_ids), block, bs), flush=True)

    res = {"vocab": tok.vocab_size, "block": block, "batch": bs,
           "train_chars": int(len(train_ids)), "val_chars": int(len(val_ids))}

    teacher = get_teacher(tok, args, res, bs, block)

    if args.mode == "teacher":
        res["val_ids"] = None
        save(res, "teacher")
        return

    if args.mode == "ablate":
        ablate(args, tok, train_ids, teacher, val_ids, bs, block, res)
        save(res, "ablate")
        return

    if args.mode == "feat":
        feat_exp(args, tok, teacher, bs, block, res)
        return

    if args.mode == "self":
        self_distill(args, tok, bs, block, res)
        return

    if args.mode == "probe":
        probe(args, tok, teacher, bs, block, res)
        return

    if args.mode == "gen":
        gen_samples(args, tok, bs, block, res)
        return

    if args.mode == "corpus_test":
        corpus_test(args, res)
        return

    if args.mode == "scratch_multi":
        scratch_multi(args, tok, bs, block, res)
        return

    if args.mode == "paired":
        paired(args, tok, bs, block, res)
        return

    if args.mode == "tscan_multi":
        tscan_multi(args, tok, bs, block, res)
        return

    if args.mode == "tuned":
        tuned(args, tok, teacher, bs, block, res)
        return

    if args.mode == "tscan":
        tscan(args, tok, bs, block, res)
        return

    if args.mode == "sweep":
        sweep(args, tok, teacher, bs, block, res)
        return

    if args.mode == "quant":
        quant_exp(args, tok, bs, block, res)
        return

    # ------------------ (2) 学生配置 ------------------
    torch.manual_seed(args.seed + 7)
    student_scratch = GPT(tok.vocab_size, STUDENT_CFG)
    torch.manual_seed(args.seed + 7)
    student_kd = GPT(tok.vocab_size, STUDENT_CFG)
    print("[student] params = %.3f M (压缩率 %.2fx)"
          % (student_scratch.n_param / 1e6, teacher.n_param / student_scratch.n_param), flush=True)
    res["student"] = {"params": student_scratch.n_param,
                      "ratio": teacher.n_param / student_scratch.n_param}

    # 对照组: 只吃硬标签
    h1, t1 = train(student_scratch, train_ids, args.student_iters, bs, block,
                   lr=args.lr, seed=args.seed + 7, teacher=None, alpha=0.0,
                   log_every=args.log_every, tag="scratch", val=val_ids)
    v1 = evaluate(student_scratch, val_ids, bs, block)
    a1 = agreement(student_scratch, teacher, val_ids, bs, block)
    res["scratch"] = {"params": student_scratch.n_param, "val_loss": v1, "ppl": math.exp(v1),
                      "agree": a1, "time_s": t1, "curve": h1, "curve_ce": student_scratch._ce_hist,
                      "iters": args.student_iters}
    print("[scratch] val_loss=%.4f ppl=%.2f agree=%.3f %.1fs"
          % (v1, math.exp(v1), a1, t1), flush=True)

    # 实验组: 同一初始化种子、同一数据顺序, 只多一项 KL
    h2, t2 = train(student_kd, train_ids, args.student_iters, bs, block,
                   lr=args.lr, seed=args.seed + 7, teacher=teacher,
                   T=args.T, alpha=args.alpha, feat_weight=args.feat,
                   log_every=args.log_every, tag="kd", val=val_ids)
    v2 = evaluate(student_kd, val_ids, bs, block)
    a2 = agreement(student_kd, teacher, val_ids, bs, block)
    res["kd"] = {"params": student_kd.n_param, "val_loss": v2, "ppl": math.exp(v2),
                 "agree": a2, "time_s": t2, "curve": h2, "curve_ce": student_kd._ce_hist,
                 "iters": args.student_iters,
                 "T": args.T, "alpha": args.alpha, "feat": args.feat}
    print("[kd] val_loss=%.4f ppl=%.2f agree=%.3f %.1fs"
          % (v2, math.exp(v2), a2, t2), flush=True)
    print("[对比] 从零 %.4f -> 蒸馏 %.4f 相对下降 %.1f%%"
          % (v1, v2, (v1 - v2) / v1 * 100), flush=True)

    torch.save(student_scratch.state_dict(), os.path.join(OUT_DIR, "student_scratch.pt"))
    torch.save(student_kd.state_dict(), os.path.join(OUT_DIR, "student_kd.pt"))

    samples = gen_samples({"teacher": teacher, "scratch": student_scratch, "kd": student_kd},
                          tok, prompt="月", n=60)
    res["samples"] = samples
    for k, v in samples.items():
        print("[gen:%s] %s" % (k, v.replace("\n", " ")), flush=True)

    plot_curves(res, os.path.join(OUT_DIR, "curves.png"))
    save(res, "all")


def ablate(args, tok, train_ids, teacher, val_ids, bs, block, res):
    """T x alpha 网格扫描。教师前向只算一次并缓存, 之后每组只训练学生。"""
    grid_T = [1, 2, 4, 8]
    grid_a = [0.3, 0.5, 0.7, 0.9]
    V = tok.vocab_size
    gen = torch.Generator().manual_seed(2024)
    cache = []
    teacher.eval()
    with torch.no_grad():
        for _ in range(32):
            x, y = get_batch(train_ids, bs, block, gen)
            cache.append((x, y, teacher(x)[0]))
    rows = []
    t_all = time.time()
    for T in grid_T:
        for a in grid_a:
            torch.manual_seed(args.seed + 7)
            m = GPT(tok.vocab_size, STUDENT_CFG)
            opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=0.01)
            m.train()
            for it in range(args.abl_iters):
                x, y, tl = cache[it % len(cache)]
                logits = m(x)[0]
                ce = F.cross_entropy(logits.view(-1, V), y.reshape(-1))
                s_lp = F.log_softmax(logits.reshape(-1, V) / T, dim=-1)
                t_p = F.softmax(tl.reshape(-1, V) / T, dim=-1)
                kd = F.kl_div(s_lp, t_p, reduction="batchmean") * (T * T)
                loss = (1 - a) * ce + a * kd
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()
            v = evaluate(m, val_ids, bs, block)
            ag = agreement(m, teacher, val_ids, bs, block)
            print("T=%d alpha=%.1f -> val %.4f ppl %.3f agree %.3f (%.0fs)"
                  % (T, a, v, math.exp(v), ag, time.time() - t_all), flush=True)
            rows.append({"T": T, "alpha": a, "val_loss": v, "ppl": math.exp(v), "agree": ag})
            res["ablation"] = rows
            save(res, "ablate")
    best = min(rows, key=lambda r: r["val_loss"])
    worst = max(rows, key=lambda r: r["val_loss"])
    print("[ablate] 最优 T=%d alpha=%.1f val=%.4f | 最差 T=%d alpha=%.1f val=%.4f"
          % (best["T"], best["alpha"], best["val_loss"],
             worst["T"], worst["alpha"], worst["val_loss"]), flush=True)
    plot_curves(res, os.path.join(OUT_DIR, "curves.png"))


def train_with_val(model, data, iters, bs, block, lr=3e-3, seed=0,
                   teacher=None, T=4.0, alpha=0.5, val=None, every=10, tag=""):
    """和 train() 同一套逻辑, 只是每 every 步在留出集上测一次, 用于画 val 曲线。"""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    gen = torch.Generator().manual_seed(seed + 1)
    if teacher is not None:
        teacher.eval()
    model.train()
    hist, valhist = [], []
    t0 = time.time()
    for it in range(iters):
        x, y = get_batch(data, bs, block, gen)
        logits = model(x)[0]
        ce = F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1))
        loss = ce
        if teacher is not None and alpha > 0:
            with torch.no_grad():
                t_logits = teacher(x)[0]
            V = logits.size(-1)
            s_lp = F.log_softmax(logits.reshape(-1, V) / T, dim=-1)
            t_p = F.softmax(t_logits.reshape(-1, V) / T, dim=-1)
            loss = (1 - alpha) * ce + alpha * F.kl_div(s_lp, t_p, reduction="batchmean") * (T * T)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        hist.append(loss.item())
        if (it + 1) % every == 0:
            valhist.append({"iter": it + 1, "val_loss": evaluate(model, val, bs, block)})
    print("[%s] 完成 %d 步, val %.4f (%.1fs)"
          % (tag, iters, valhist[-1]["val_loss"], time.time() - t0), flush=True)
    return hist, valhist, time.time() - t0


def gen_samples(args, tok, bs, block, res):
    """用固定采样种子生成三组样例, 保证三模型可比"""
    import torch.nn.functional as Fn
    def sample(m, prompt, n=60, temp=0.8, top_k=40, seed=2027):
        m.eval()
        ids = torch.tensor([tok.encode(prompt)], dtype=torch.long)
        gen = torch.Generator().manual_seed(seed)
        out = list(ids[0].tolist())
        ctx = ids
        for _ in range(n):
            ctx = ctx[:, -block:]
            logits = m(ctx)[0][:, -1, :] / temp
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")
            probs = Fn.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1, generator=gen)
            out.append(int(nxt))
            ctx = torch.cat([ctx, nxt], dim=1)
        return tok.decode(out)

    rows = []
    for name, f in (("teacher", "teacher.pt"), ("scratch", "student_scratch.pt"),
                    ("kd", "student_kd.pt")):
        path = os.path.join(OUT_DIR, f)
        if not os.path.exists(path):
            print("[gen] 缺", f); continue
        cfg = TEACHER_CFG if name == "teacher" else STUDENT_CFG
        m = GPT(tok.vocab_size, cfg)
        m.load_state_dict(torch.load(path, map_location="cpu"))
        txt = sample(m, "月")
        rows.append({"model": name, "text": txt})
        print("[gen] %-8s %s" % (name, txt), flush=True)
    res["gen"] = {"prompt": "月", "temp": 0.8, "top_k": 40, "seed": 2027, "runs": rows}
    save(res, "gen")


def build_corpus_b(n_lines=3000, seed=0):
    """语料 B: 标点复原成指南风格(全角逗号/句号/冒号引号), 用于检验"语料标点是否影响蒸馏收益符号" """
    rng = random.Random(seed)
    out = []
    for i in range(n_lines):
        s_, p_, v_, o_, m_ = (rng.choice(x) for x in (SUBJ, PLACE, VERB, OBJ, MOOD))
        if i % 3 == 0:
            out.append("%s%s%s%s，%s。" % (s_, p_, v_, o_, m_))
        elif i % 3 == 1:
            out.append("%s在%s了一夜%s，%s。" % (s_, p_, o_, m_))
        else:
            out.append("%s%s道：“%s%s。”%s。" % (s_, v_, o_, rng.choice(["正好", "可怜", "可惜", "依旧"]), m_))
    poems = [x.replace(", ", "，").replace("。", "。").replace(",", "，") for x in POEMS]
    out.extend(poems)
    return "\n".join(out)


def corpus_test(args, res):
    """同一套训练/评测流程, 只换语料标点体系, 看蒸馏收益的符号是否改变"""
    raw_tr = build_corpus_b(3000, seed=0)
    raw_va = build_corpus_b(300, seed=999)
    tok_b = Tokenizer(raw_tr + raw_va)
    tr = torch.tensor(tok_b.encode(raw_tr), dtype=torch.long)
    va = torch.tensor(tok_b.encode(raw_va), dtype=torch.long)
    print("[corpusB] vocab=%d train_chars=%d val_chars=%d"
          % (tok_b.vocab_size, tr.numel(), va.numel()), flush=True)
    bs, block = 24, 96
    torch.manual_seed(args.seed + 7)
    teacher = GPT(tok_b.vocab_size, TEACHER_CFG)
    _, ts = train(teacher, tr, args.teacher_iters, bs, block, lr=args.lr,
                  seed=args.seed + 7, log_every=999, tag="corpusB_teacher")
    tv = evaluate(teacher, va, bs, block)
    print("[corpusB] teacher val %.4f ppl %.2f (%.1fs)" % (tv, math.exp(tv), ts), flush=True)
    rows = []
    for name, T, a in (("scratch", 4.0, 0.0), ("kd_default", 4.0, 0.5), ("kd_tuned", 4.0, 0.3)):
        torch.manual_seed(args.seed + 7)
        m = GPT(tok_b.vocab_size, STUDENT_CFG)
        _, ts2 = train(m, tr, args.student_iters, bs, block, lr=args.lr, seed=args.seed + 7,
                       teacher=None if a == 0 else teacher, T=T, alpha=a,
                       log_every=999, tag="corpusB_" + name)
        v = evaluate(m, va, bs, block)
        ag = agreement(m, teacher, va, bs, block)
        rows.append({"name": name, "T": T, "alpha": a, "val_loss": v,
                     "ppl": math.exp(v), "agree": ag, "time_s": ts2})
        print("[corpusB] %-11s val %.4f ppl %.2f agree %.3f" % (name, v, math.exp(v), ag), flush=True)
    res["corpusB"] = {"vocab": tok_b.vocab_size, "train_chars": int(tr.numel()),
                      "val_chars": int(va.numel()), "teacher_val": tv, "runs": rows}
    save(res, "corpusB")


def scratch_multi(args, tok, bs, block, res):
    """从零学生的多种子基线, 供教师规模扫描做同基准比较"""
    rows = []
    for sd in (7, 17, 27):
        torch.manual_seed(sd)
        m = GPT(tok.vocab_size, STUDENT_CFG)
        _, ts = train(m, train_ids_global, args.student_iters, bs, block, lr=args.lr,
                      seed=sd, alpha=0.0, log_every=999, tag="sc%d" % sd)
        v = evaluate(m, val_ids_global, bs, block)
        rows.append({"seed": sd, "val_loss": v, "time_s": ts})
        print("[scratch_multi] seed=%2d -> %.4f" % (sd, v), flush=True)
    mean = sum(r["val_loss"] for r in rows) / len(rows)
    res["scratch_multi"] = {"runs": rows, "mean": mean,
                            "spread": max(r["val_loss"] for r in rows) - min(r["val_loss"] for r in rows)}
    print("[scratch_multi] 均值 %.4f 极差 %.3f" % (mean, res["scratch_multi"]["spread"]), flush=True)
    save(res, "scratch_multi")


def paired(args, tok, bs, block, res):
    """配对评测: 同一评测批次下比较两组模型的 loss 差, 排除评测噪声"""
    items = [("scratch", "student_scratch.pt", dict(STUDENT_CFG)),
             ("kd_default", "student_kd.pt", dict(STUDENT_CFG)),
             ("kd_tuned", "student_tuned.pt", dict(STUDENT_CFG)),
             ("kd_T8_a0.1", "probe_8_0.1.pt", dict(STUDENT_CFG))]
    models = {}
    for name, f, cfg in items:
        p = os.path.join(OUT_DIR, f)
        if not os.path.exists(p):
            print("[paired] 缺权重, 跳过", f); continue
        m = GPT(tok.vocab_size, cfg)
        m.load_state_dict(torch.load(p, map_location="cpu"))
        models[name] = m
    seeds = [1000 + i for i in range(10)]
    per = {}
    for name, m in models.items():
        per[name] = [evaluate(m, val_ids_global, bs, block, seed=s) for s in seeds]
    ref = per["scratch"]
    out = {"seeds": seeds, "abs": {}, "delta": {}}
    for name, vals in per.items():
        mu = sum(vals) / len(vals)
        sd = (sum((x - mu) ** 2 for x in vals) / (len(vals) - 1)) ** 0.5
        out["abs"][name] = {"vals": vals, "mean": mu, "std": sd}
        d = [v - r for v, r in zip(vals, ref)]
        dm = sum(d) / len(d)
        ds = (sum((x - dm) ** 2 for x in d) / (len(d) - 1)) ** 0.5
        out["delta"][name] = {"deltas": d, "mean": dm, "std": ds}
        print("[paired] %-12s 绝对 %.4f±%.4f | 相对 scratch 差 %+.4f±%.4f"
              % (name, mu, sd, dm, ds), flush=True)
    res["paired"] = out
    save(res, "paired")


def probe(args, tok, teacher, bs, block, res):
    """120 步下的超参探针: 固定步数, 与主实验可比; 并测评测噪声"""
    cfgs = [(4.0, 0.0), (2.0, 0.3), (4.0, 0.3), (4.0, 0.5),
            (8.0, 0.1), (8.0, 0.3), (8.0, 0.5)]
    rows = []
    for T, a in cfgs:
        torch.manual_seed(args.seed + 7)
        m = GPT(tok.vocab_size, STUDENT_CFG)
        use_t = None if a == 0 else teacher
        _, ts = train(m, train_ids_global, args.student_iters, bs, block, lr=args.lr,
                      seed=args.seed + 7, teacher=use_t, T=T, alpha=a,
                      log_every=999, tag="probe_%.0f_%.1f" % (T, a))
        v = evaluate(m, val_ids_global, bs, block)
        ag = agreement(m, teacher, val_ids_global, bs, block)
        torch.save(m.state_dict(), os.path.join(OUT_DIR, "probe_%.0f_%.1f.pt" % (T, a)))
        rows.append({"T": T, "alpha": a, "val_loss": v, "ppl": math.exp(v),
                     "agree": ag, "time_s": ts})
        print("[probe] T=%.0f a=%.1f -> val %.4f ppl %.3f agree %.3f (%.1fs)"
              % (T, a, v, math.exp(v), ag, ts), flush=True)
    res["probe"] = rows

    # 评测噪声: 同一模型、换评测种子重复测, 看 loss 的波动幅度
    path = os.path.join(OUT_DIR, "student_kd.pt")
    if os.path.exists(path):
        m = GPT(tok.vocab_size, STUDENT_CFG)
        m.load_state_dict(torch.load(path, map_location="cpu"))
        ss = []
        for sd in range(10):
            v = evaluate(m, val_ids_global, bs, block, seed=1000 + sd)
            ss.append(v)
        res["eval_noise"] = {"seed": 1000, "n": len(ss), "vals": ss,
                             "mean": sum(ss) / len(ss),
                             "min": min(ss), "max": max(ss)}
        print("[probe] 评测噪声: 同一模型 10 次评测 %.4f~%.4f, 均值 %.4f"
              % (min(ss), max(ss), sum(ss) / len(ss)), flush=True)
    save(res, "probe")


def tscan_multi(args, tok, bs, block, res):
    """教师规模扫描的多种子版本: 每个教师配 3 个学生初始化种子"""
    seeds = (7, 17, 27)
    rows = []
    for C in (96, 160, 256):
        cfg = dict(n_layer=2, n_head=4, n_embd=C, block_size=96, dropout=0.0)
        path = os.path.join(OUT_DIR, "teacher_c%d.pt" % C) if C != 160 else os.path.join(OUT_DIR, "teacher.pt")
        t = GPT(tok.vocab_size, cfg)
        t.load_state_dict(torch.load(path, map_location="cpu"))
        vt = evaluate(t, val_ids_global, bs, block)
        vals = []
        for sd in seeds:
            torch.manual_seed(sd)
            m = GPT(tok.vocab_size, STUDENT_CFG)
            _, ts = train(m, train_ids_global, args.student_iters, bs, block, lr=args.lr,
                          seed=sd, teacher=t, T=8.0, alpha=0.3,
                          log_every=999, tag="ts%d_%d" % (C, sd))
            v = evaluate(m, val_ids_global, bs, block)
            vals.append(v)
            print("[tscan_multi] C=%d seed=%2d -> student %.4f" % (C, sd, v), flush=True)
        mean = sum(vals) / len(vals)
        spread = max(vals) - min(vals)
        rows.append({"C": C, "params": t.n_param, "teacher_val": vt,
                     "students": vals, "mean": mean, "spread": spread,
                     "seeds": list(seeds)})
        print("[tscan_multi] C=%d 教师 %.4f | 学生 %.4f~%.4f 均值 %.4f"
              % (C, vt, min(vals), max(vals), mean), flush=True)
    res["tscan_multi"] = rows
    save(res, "tscan_multi")


def tuned(args, tok, teacher, bs, block, res):
    """按消融最优点(T=4, alpha=0.3)复跑一组学生, 与默认配置对照"""
    torch.manual_seed(args.seed + 7)
    m = GPT(tok.vocab_size, STUDENT_CFG)
    h, t = train(m, train_ids_global, args.student_iters, bs, block, lr=args.lr,
                 seed=args.seed + 7, teacher=teacher, T=args.T, alpha=args.alpha,
                 log_every=args.log_every, tag="tuned", val=val_ids_global)
    v = evaluate(m, val_ids_global, bs, block)
    ag = agreement(m, teacher, val_ids_global, bs, block)
    gen = gen_samples({"tuned": m}, tok, prompt="月", n=60)
    res["tuned"] = {"T": args.T, "alpha": args.alpha, "val_loss": v, "ppl": math.exp(v),
                    "agree": ag, "time_s": t, "curve": h, "curve_ce": m._ce_hist,
                    "samples": gen}
    print("[tuned] T=%.0f alpha=%.1f val_loss=%.4f ppl=%.3f agree=%.3f %.1fs"
          % (args.T, args.alpha, v, math.exp(v), ag, t), flush=True)
    torch.save(m.state_dict(), os.path.join(OUT_DIR, "student_tuned.pt"))
    save(res, "tuned")


def tscan(args, tok, bs, block, res):
    """拓展 E: 教师规模扫描 C in {96,160,256}, 看蒸馏收益是否随教师变强而增大"""
    rows = []
    for C in (96, 160, 256):
        cfg = dict(n_layer=2, n_head=4, n_embd=C, block_size=96, dropout=0.0)
        if C == 160:
            t = GPT(tok.vocab_size, cfg)
            t.load_state_dict(torch.load(os.path.join(OUT_DIR, "teacher.pt"), map_location="cpu"))
            tt = 0.0
        else:
            torch.manual_seed(args.seed)
            t = GPT(tok.vocab_size, cfg)
            _, tt = train(t, train_ids_global, args.teacher_iters, bs, block, lr=args.lr,
                          seed=args.seed, log_every=999, tag="t%d" % C, val=val_ids_global)
            torch.save(t.state_dict(), os.path.join(OUT_DIR, "teacher_c%d.pt" % C))
        vt = evaluate(t, val_ids_global, bs, block)
        row = {"C": C, "params": t.n_param, "teacher_val": vt,
               "ppl": math.exp(vt), "teacher_time_s": tt, "students": {}}
        for conf in ((4.0, 0.5), (8.0, 0.3)):
            torch.manual_seed(args.seed + 7)
            m = GPT(tok.vocab_size, STUDENT_CFG)
            _, ts = train(m, train_ids_global, args.student_iters, bs, block, lr=args.lr,
                          seed=args.seed + 7, teacher=t, T=conf[0], alpha=conf[1],
                          log_every=999, tag="s%d_%.0f_%.1f" % (C, conf[0], conf[1]))
            v = evaluate(m, val_ids_global, bs, block)
            ag = agreement(m, t, val_ids_global, bs, block)
            row["students"]["T%.0f_a%.1f" % conf] = {"val_loss": v, "ppl": math.exp(v),
                                                     "agree": ag, "time_s": ts}
            print("[tscan] C=%d teacher %.4f | T=%.0f a=%.1f -> student %.4f (%.1fs)"
                  % (C, vt, conf[0], conf[1], v, ts), flush=True)
        rows.append(row)
        res["tscan"] = rows
        save(res, "tscan")


def sweep(args, tok, teacher, bs, block, res):
    """拓展/分析: 学生训练步数扫描, 看蒸馏收益随欠拟合->过拟合的迁移"""
    out = {}
    for tag, use_t in (("scratch", False), ("kd", True)):
        torch.manual_seed(args.seed + 7)
        m = GPT(tok.vocab_size, STUDENT_CFG)
        hist, vh, tt = train_with_val(
            m, train_ids_global, args.teacher_iters, bs, block, lr=args.lr,
            seed=args.seed + 7, teacher=teacher if use_t else None,
            T=args.T, alpha=args.alpha, val=val_ids_global, every=10, tag=tag)
        out[tag] = {"val_hist": vh, "time_s": tt, "curve": hist}
        if use_t:
            torch.save(m.state_dict(), os.path.join(OUT_DIR, "student_kd_sweep.pt"))
    # 对比表
    tab = []
    for a, b in zip(out["scratch"]["val_hist"], out["kd"]["val_hist"]):
        tab.append({"iter": a["iter"], "scratch": a["val_loss"], "kd": b["val_loss"],
                    "rel_gain_%": (a["val_loss"] - b["val_loss"]) / a["val_loss"] * 100})
    res["sweep"] = {"runs": out, "table": tab}
    for r in tab:
        print("steps %3d | scratch %.4f | kd %.4f | kd 相对变化 %+.1f%%"
              % (r["iter"], r["scratch"], r["kd"], r["rel_gain_%"]), flush=True)
    save(res, "sweep")


def feat_exp(args, tok, teacher, bs, block, res):
    """拓展 A: 特征蒸馏 (FitNet), 对比 feat=0 与 feat=0.1"""
    out = {}
    for w in (0.0, 0.1):
        torch.manual_seed(args.seed + 7)
        m = GPT(tok.vocab_size, STUDENT_CFG)
        h, t = train(m, train_ids_global, args.student_iters, bs, block,
                     lr=args.lr, seed=args.seed + 7, teacher=teacher,
                     T=args.T, alpha=args.alpha, feat_weight=w,
                     log_every=args.log_every, tag="feat%.1f" % w, val=val_ids_global)
        v = evaluate(m, val_ids_global, bs, block)
        ag = agreement(m, teacher, val_ids_global, bs, block)
        out["feat_%.1f" % w] = {"val_loss": v, "ppl": math.exp(v), "agree": ag,
                                "time_s": t, "curve": h}
        print("[feat=%.1f] val_loss=%.4f ppl=%.2f agree=%.3f (%.0fs)"
              % (w, v, math.exp(v), ag, t), flush=True)
    res["feat"] = out
    g = gen_samples({"kd": m}, tok, prompt="月", n=60)
    res["feat_gen"] = g
    save(res, "feat")


def self_distill(args, tok, bs, block, res):
    """拓展 C: 自蒸馏——学生同结构, 但先训练 3 倍步数作为「教师」"""
    big_iters = args.student_iters * 3
    torch.manual_seed(args.seed + 7)
    t_student = GPT(tok.vocab_size, STUDENT_CFG)
    h, tt = train(t_student, train_ids_global, big_iters, bs, block, lr=args.lr,
                  seed=args.seed + 7, log_every=args.log_every, tag="sd-teacher",
                  val=val_ids_global)
    vt = evaluate(t_student, val_ids_global, bs, block)
    print("[sd-teacher] val_loss=%.4f ppl=%.2f (%.0fs)" % (vt, math.exp(vt), tt), flush=True)

    torch.manual_seed(args.seed + 7)
    s_scratch = GPT(tok.vocab_size, STUDENT_CFG)
    h1, t1 = train(s_scratch, train_ids_global, args.student_iters, bs, block, lr=args.lr,
                   seed=args.seed + 7, log_every=args.log_every, tag="sd-scratch",
                   val=val_ids_global)
    v1 = evaluate(s_scratch, val_ids_global, bs, block)

    torch.manual_seed(args.seed + 7)
    s_kd = GPT(tok.vocab_size, STUDENT_CFG)
    h2, t2 = train(s_kd, train_ids_global, args.student_iters, bs, block, lr=args.lr,
                   seed=args.seed + 7, teacher=t_student, T=args.T, alpha=args.alpha,
                   log_every=args.log_every, tag="sd-kd", val=val_ids_global)
    v2 = evaluate(s_kd, val_ids_global, bs, block)
    print("[self-distill] scratch %.4f | kd %.4f | 相对下降 %.1f%%"
          % (v1, v2, (v1 - v2) / v1 * 100), flush=True)
    res["self_distill"] = {
        "teacher_iters": big_iters, "teacher_val": vt, "teacher_time_s": tt,
        "scratch": {"val_loss": v1, "ppl": math.exp(v1), "time_s": t1, "curve": h1},
        "kd": {"val_loss": v2, "ppl": math.exp(v2), "time_s": t2, "curve": h2},
    }
    samples = gen_samples({"sd_teacher": t_student, "sd_scratch": s_scratch, "sd_kd": s_kd},
                          tok, prompt="月", n=60)
    res["self_distill"]["samples"] = samples
    save(res, "self")


def quant_exp(args, tok, bs, block, res):
    """拓展 D: 对蒸馏得到的学生做动态量化 (int8)"""
    path = os.path.join(OUT_DIR, "student_kd.pt")
    if not os.path.exists(path):
        print("[quant] 先跑 --mode all 生成 student_kd.pt", flush=True)
        return
    m = GPT(tok.vocab_size, STUDENT_CFG)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    v_fp32 = evaluate(m, val_ids_global, bs, block)

    fp32_path = os.path.join(OUT_DIR, "student_fp32.pt")
    torch.save(m.state_dict(), fp32_path)
    size_fp32 = os.path.getsize(fp32_path)

    q = torch.quantization.quantize_dynamic(m, {nn.Linear}, dtype=torch.qint8)
    int8_path = os.path.join(OUT_DIR, "student_int8.pt")
    torch.save(q.state_dict(), int8_path)
    size_int8 = os.path.getsize(int8_path)

    v_int8 = evaluate(q, val_ids_global, bs, block)
    ag_int8 = agreement(q, m, val_ids_global, bs, block)

    # 单次前向延迟 (batch=1, block=96), 各跑 30 次取均值
    x = torch.randint(0, tok.vocab_size, (1, block))

    def bench(model, n=30):
        model.eval()
        with torch.no_grad():
            model(x)
            t0 = time.time()
            for _ in range(n):
                model(x)
        return (time.time() - t0) / n * 1000

    lat_fp32 = bench(m)
    lat_int8 = bench(q)

    print("[quant] fp32 %.1f KB val %.4f | %.1f ms"
          % (size_fp32 / 1024, v_fp32, lat_fp32), flush=True)
    print("[quant] int8 %.1f KB val %.4f | %.1f ms | 与fp32一致率 %.3f"
          % (size_int8 / 1024, v_int8, lat_int8, ag_int8), flush=True)
    res["quant"] = {"size_fp32": size_fp32, "size_int8": size_int8,
                    "ratio": size_fp32 / size_int8, "val_fp32": v_fp32,
                    "val_int8": v_int8, "ppl_fp32": math.exp(v_fp32),
                    "ppl_int8": math.exp(v_int8), "lat_fp32_ms": lat_fp32,
                    "lat_int8_ms": lat_int8, "agree_int8": ag_int8}
    g = gen_samples({"int8": q}, tok, prompt="月", n=60)
    res["quant"]["gen"] = g
    save(res, "quant")


if __name__ == "__main__":
    main()
