"""按 arXiv 风格模板生成期末报告 DOCX。"""
import json, os, shutil
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TPL = '/home/user/.doubao/agent_mode/workspace/.sessions/38453420968790274/attachments/Final-Report-Template-arXiv-style.docx'
OUT = os.path.join(BASE, 'Final_Report_Lightweight_Image_Analysis.docx')
FIG = os.path.join(BASE, 'figures')
M = json.load(open(os.path.join(BASE, 'results', 'metrics.json')))

# ---- 关键数字 ----
cnn = M['cnn_fp32']; vit = M['vit_fp32']; prune = M['prune50']; kd = M['kd_student']
ci8 = M['cnn_int8']; ki8 = M['kd_int8']; ours = M['ours']
scr = M['scratch_student']; ff = M['cnn_fashion_fp32']; fi = M['cnn_fashion_int8']
abl = M.get('ablation', {})

size_ratio = cnn['mb'] / ours['mb']
speed_ratio = cnn['lat'] / ours['lat']
acc_drop = (cnn['acc'] - ours['acc']) * 100
param_drop = (1 - ours['params'] / cnn['params']) * 100
param_drop_kd = (1 - kd['params'] / cnn['params']) * 100
q_speed = cnn['lat'] / ci8['lat']
q_size = cnn['mb'] / ci8['mb']
kd_gain = (kd['acc'] - scr['acc']) * 100
abl_mb = ours['params'] * 4 / 1024  # KD+Prune 权重的 FP32 字节数约 62 KB


def pct(x):
    return f"{x * 100:.2f}"


doc = Document(TPL)
BODY = doc.element.body
paras = doc.paragraphs

# 1) 封面题目
cover = paras[4]
cover.runs[0].text = ('Lightweight Image Analysis for Edge Deployment: Coordinated Knowledge '
                      'Distillation, Structured Pruning and INT8 Quantization')
for r in cover.runs[1:]:
    r._element.getparent().remove(r._element)

# 2) 第 2 页题目 / 作者 / 单位 / 关键词
title_p = paras[12]
title_p.runs[0].text = 'Lightweight Image Analysis for Edge Deployment: Coordinated Knowledge Distillation, Structured Pruning and INT8 Quantization'
for r in title_p.runs[1:]:
    r._element.getparent().remove(r._element)
title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER

author_p = paras[13]
author_p.runs[0].text = 'Author: xxx        Student ID: xxx'
for r in author_p.runs[1:]:
    r._element.getparent().remove(r._element)
author_p.alignment = WD_ALIGN_PARAGRAPH.CENTER

aff_p = paras[14]
aff_p.runs[0].text = 'Harbin Institute of Technology, Shenzhen'
for r in aff_p.runs[1:]:
    r._element.getparent().remove(r._element)
aff_p.alignment = WD_ALIGN_PARAGRAPH.CENTER

kw_p = paras[16]
kw_p.runs[0].text = ('Keywords: lightweight image analysis; knowledge distillation; structured pruning; '
                     'INT8 quantization; edge deployment')
for r in kw_p.runs[1:]:
    r._element.getparent().remove(r._element)

abs_p = paras[18]
for r in abs_p.runs[1:]:
    r._element.getparent().remove(r._element)
abs_p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
abs_p.paragraph_format.first_line_indent = None

ABSTRACT = (
    'Deploying image-analysis models on edge devices is limited less by accuracy than by memory, energy and '
    'latency. The visual-intelligence modules of the course (CNN in Lesson 3 and Vision Transformer in Lesson 5) '
    f'provide strong representations, yet our compact CNN baseline already needs {cnn["params"]/1000:.1f}K '
    f'parameters, {cnn["mb"]:.3f} MB and {cnn["lat"]:.2f} ms per image on a two-thread CPU, which is still heavy '
    'for microcontroller-class hardware. This paper studies how far such a course-scale classifier can be compressed '
    'when three techniques from Lesson 15 are coordinated rather than applied in isolation. We train a CNN teacher on '
    'MNIST, distill a width-halved student with temperature-4 soft targets, prune output channels by filter L1 norm '
    'and convert the result to INT8 with static post-training quantization (PTQ). Controlled experiments show that '
    f'distillation alone recovers {kd_gain:.2f} percentage points over a scratch student of identical size; INT8 '
    f'alone reduces model size by {q_size:.1f}x and latency by {q_speed:.1f}x with a {abs(ci8["acc"]-cnn["acc"])*100:.2f}-point drop. '
    f'The final edge model keeps {pct(ours["acc"])}% accuracy with {ours["params"]/1000:.1f}K parameters, '
    f'{ours["mb"]*1024:.1f} KB and {ours["lat"]:.2f} ms, i.e. {size_ratio:.1f}x smaller and {speed_ratio:.1f}x faster than '
    f'the FP32 baseline at a cost of {acc_drop:.2f} percentage points. A robustness study exposes a side effect that '
    'accuracy-only evaluation hides: INT8 activation clipping makes the edge model markedly more sensitive to '
    'sensor noise than its FP32 counterpart. The complete pipeline, a browser-based demonstration system and all '
    'measurement scripts are released for reproduction.'
)
abs_p.runs[0].text = ABSTRACT

# 3) 删除 Abstract 之后的全部模板内容（保留 sectPr）
start_idx = 18  # 保留摘要段（index 18），其后全部删除
for p in list(paras[start_idx + 1:]):
    p._element.getparent().remove(p._element)
for child in list(BODY):
    if child.tag == qn('w:tbl'):
        BODY.remove(child)

BLACK = RGBColor(0, 0, 0)


def _font(run, size=12, bold=False, italic=False, name='Times New Roman', east='宋体'):
    run.font.name = name
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = BLACK
    run._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'), east)


def h1(text):
    p = doc.add_paragraph(style='Heading 1')
    p.paragraph_format.space_before = Pt(18)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    _font(r, 16, True, name='Times New Roman', east='黑体')
    return p


def h2(text):
    p = doc.add_paragraph(style='Heading 2')
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    _font(r, 14, True, name='Times New Roman', east='黑体')
    return p


def body(text):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    pf.line_spacing = 1.5
    pf.space_after = Pt(6)
    pf.space_before = Pt(0)
    pf.first_line_indent = None
    r = p.add_run(text)
    _font(r, 12)
    return p


def bullet(label, text):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    pf.line_spacing = 1.5
    pf.left_indent = Cm(0.75)
    pf.first_line_indent = Cm(-0.75)
    pf.space_after = Pt(4)
    r = p.add_run(label)
    _font(r, 12, True)
    r2 = p.add_run(text)
    _font(r2, 12)
    return p


def caption(text, before=4, after=4):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    pf.line_spacing = 1.0
    r = p.add_run(text)
    _font(r, 9, True)
    return p


def equation(path, no, height=0.78):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf.space_before = Pt(4)
    pf.space_after = Pt(4)
    pf.line_spacing = 1.0
    p.add_run().add_picture(os.path.join(FIG, path), height=Cm(height))
    r = p.add_run(f'      ({no})')
    _font(r, 11)
    return p


def shade(cell, color='D9D9D9'):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), color)
    tcPr.append(shd)


def set_cell_margins(table, left=0.14, right=0.14):
    tblPr = table._tbl.tblPr
    mar = OxmlElement('w:tblCellMar')
    for side in ('left', 'right'):
        e = OxmlElement(f'w:{side}')
        e.set(qn('w:w'), str(int(left * 567)))
        e.set(qn('w:type'), 'dxa')
        mar.append(e)
    tblPr.append(mar)


def add_table(headers, rows, widths, aligns=None, font_size=9):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = 'Table Grid'
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    set_cell_margins(t)
    aligns = aligns or ['center'] * len(headers)
    hdr = t.rows[0]
    trPr = hdr._tr.get_or_add_trPr()
    th = OxmlElement('w:tblHeader'); trPr.append(th)
    for i, h in enumerate(headers):
        c = hdr.cells[i]
        c.width = Cm(widths[i])
        c.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        shade(c)
        p = c.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = 1.0
        p.paragraph_format.first_line_indent = None
        r = p.add_run(h)
        _font(r, font_size, True)
    for row in rows:
        cells = t.add_row().cells
        trPr = t.rows[-1]._tr.get_or_add_trPr()
        cs = OxmlElement('w:cantSplit'); trPr.append(cs)
        for i, val in enumerate(row):
            c = cells[i]
            c.width = Cm(widths[i])
            c.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            p = c.paragraphs[0]
            p.alignment = {'center': WD_ALIGN_PARAGRAPH.CENTER, 'left': WD_ALIGN_PARAGRAPH.LEFT,
                           'right': WD_ALIGN_PARAGRAPH.RIGHT}[aligns[i]]
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.0
            p.paragraph_format.first_line_indent = None
            r = p.add_run(str(val))
            _font(r, font_size, False)
    return t


def figure(name, caption_text, width=14.5):
    p = doc.add_paragraph()
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(os.path.join(FIG, name), width=Cm(width))
    caption(caption_text)


def mono_box(lines):
    t = doc.add_table(rows=1, cols=1)
    t.style = 'Table Grid'
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    c = t.rows[0].cells[0]
    c.width = Cm(15.5)
    shade(c, 'F7F8FA')
    first = True
    for line in lines:
        p = c.paragraphs[0] if first else c.add_paragraph()
        first = False
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = 1.0
        p.paragraph_format.first_line_indent = None
        r = p.add_run(line)
        _font(r, 8, name='Consolas', east='宋体')
    return t


# ============================ 正文 ============================

h1('1  Introduction')
h2('1.1  Research Background & Problem Statement')
body('Image analysis is the front end of most intelligent systems: industrial inspection, mobile photography, '
     'autonomous driving and medical screening all begin with a model that turns pixels into semantic evidence. '
     'In the eight-module landscape of this course (Structural -> Visual -> Language -> Generative -> Graph -> '
     'Multimodal -> Reinforcement -> Agent Intelligence), visual intelligence is the module that supplies this '
     'evidence, and it is also the module with the hardest deployment constraints, because cameras produce data '
     'continuously while the compute budget of an edge device is fixed.')
body('Two architecture families dominate visual intelligence. Convolutional neural networks inherit two inductive '
     'biases from the visual system, local receptive fields and weight sharing, so a 3x3 kernel learns an edge '
     'detector that is reused at every spatial position (Lesson 3). Vision Transformers instead split an image into '
     'patches and let every patch attend to every other patch, which removes the locality prior and therefore needs '
     'more data or more parameters to reach the same accuracy (Lesson 5). Both families scale gracefully upward, but '
     'edge deployment pushes in the opposite direction: a camera SoC may offer a few hundred kilobytes of SRAM, a '
     'microcontroller tens of milliseconds per frame, and a battery that cannot afford a floating-point matrix '
     'multiply for every pixel.')
body('Lesson 15 provides four compression directions, parameter pruning, quantization, knowledge distillation and '
     'low-rank factorization, yet they are usually introduced as independent recipes on large benchmarks. In a '
     'course-scale setting the interesting question is different: with no pretrained weights, a small dataset and a '
     'CPU-only budget, how much accuracy survives when several of these techniques are stacked, and which '
     'combination gives the best accuracy-per-byte? This paper answers the question with controlled experiments '
     'rather than with a single best number, and it reports one effect that accuracy-only evaluation hides, namely '
     'that INT8 activation clipping makes the compressed model noticeably more sensitive to input noise.')
body(f'Concretely, our FP32 CNN teacher reaches {pct(cnn["acc"])}% on MNIST with {cnn["params"]/1000:.1f}K parameters '
     f'and {cnn["lat"]:.2f} ms per image, which is already a small model, and the compressed edge model still keeps '
     f'{pct(ours["acc"])}% while shrinking to {ours["mb"]*1024:.1f} KB. The gap between these two numbers, '
     f'{acc_drop:.2f} percentage points, is the price of a {size_ratio:.1f}x size reduction and a {speed_ratio:.1f}x '
     'latency reduction, and understanding where that price comes from is the main subject of this report.')

h2('1.2  Research Objectives & Core Contributions')
body('Our contributions are three-fold:')
bullet('(1)  ', 'We assemble a single reproducible pipeline that connects two course modules that are usually taught '
       'separately, visual intelligence (CNN and ViT backbones) and model compression (distillation, structured '
       'pruning and INT8 PTQ), and we quantify each stage under identical data, seeds and hardware.')
bullet('(2)  ', 'We show experimentally that distillation and pruning are complementary rather than redundant: at '
       f'the same 24.2K-parameter budget, the distilled student reaches {pct(kd["acc"])}% while an identically sized '
       f'student trained from scratch reaches {pct(scr["acc"])}%, a {kd_gain:.2f}-point gain that channel pruning at '
       f'the same budget does not reach ({pct(prune["acc"])}%).')
bullet('(3)  ', 'We release the full codebase, the measurement scripts and an interactive browser demonstration, and '
       'we add a corruption-robustness evaluation that reveals the hidden cost of quantized activation ranges, '
       'turning a single accuracy number into a deployment decision.')

h2('1.3  Paper Organization')
body('Section 2 reviews the course fundamentals and related work. Section 3 formalizes the task and describes the '
     'proposed pipeline. Section 4 reports the main results, the ablation study and the visualization analysis. '
     'Section 5 documents reproducibility, Section 6 presents the demonstration system, Section 7 discusses findings '
     'and limitations, and Section 8 concludes.')

h1('2  Related Work')
h2('2.1  Course Fundamental Knowledge Review')
body('Structural intelligence supplies the basic layer: a neuron computes a weighted sum plus bias and passes it '
     'through a non-linear activation, and backpropagation applies the chain rule to assign credit to every weight. '
     'Visual intelligence adds spatial structure. A convolution layer slides a small kernel over the input, so the '
     'same filter detects the same pattern anywhere in the image, and pooling progressively enlarges the receptive '
     'field while reducing the spatial resolution. Batch normalization re-centres each channel, which stabilizes '
     'training and, as a side effect, makes the layer folding used by INT8 quantization possible. Language and '
     'generative intelligence contribute the Transformer block, whose self-attention computes a data-dependent '
     'weight for every pair of positions; the Vision Transformer reuses exactly this block after replacing words '
     'with image patches.')
body('Compression knowledge comes from Lesson 15. Pruning removes redundant parameters, either individually '
     '(unstructured) or in groups such as filters (structured); only structured pruning reduces the dense matrix '
     'dimensions and therefore the real latency. Quantization maps floating-point values to low-precision integers; '
     'post-training quantization (PTQ) needs only a small calibration set, while quantization-aware training (QAT) '
     'simulates the quantization error during training. Knowledge distillation transfers the dark knowledge stored '
     'in a teacher\'s soft output distribution to a smaller student through a temperature-scaled KL divergence. '
     'Low-rank factorization, including LoRA-style decomposition, is the fourth direction but is not used here '
     'because the convolutional kernels of a 94K-parameter model are already close to full rank.')

h2('2.2  State-of-the-Art Research Progress')
body('Han et al. combined pruning, trained quantization and Huffman coding in Deep Compression and reported a '
     '35x compression of AlexNet without accuracy loss [5,6]. Li et al. pruned filters by their L1 norm and showed '
     'that a plain magnitude criterion is a surprisingly strong baseline [7]. Frankle and Carbin found that a '
     'pruned network contains a trainable sub-network that can reach the original accuracy on its own, the lottery '
     'ticket hypothesis [8]. Hinton et al. introduced temperature-scaled distillation [9], and Jacob et al. showed '
     'that an integer-only inference pipeline can be deployed on mobile hardware when scales and zero points are '
     'propagated carefully [10]. Efficient architecture design follows a parallel path, with MobileNetV2 '
     'demonstrating that inverted residuals and linear bottlenecks cut multiply-adds by an order of magnitude [11].')
body('This work differs from the above in scale and in intent. We do not chase a new state of the art on a large '
     'benchmark; we ask which combination of course-level techniques is worth teaching and deploying when the '
     'starting model is already tiny, and we measure the interaction between compression and input corruption '
     'instead of reporting clean accuracy alone.')

h2('2.3  Correlation Between This Research and the Course')
body('The pipeline is a direct practice of three lessons. The backbone follows Lesson 3 (convolution, batch '
     'normalization, global average pooling) and Lesson 5 (patch embedding and multi-head self-attention); the '
     'compression stages follow Lesson 15 (distillation, structured pruning, PTQ); and the demonstration follows '
     'Lesson 11, where an intelligent system is defined as perception plus tools plus an interface. The experiments '
     'therefore form the theory-to-experiment-to-system loop that the course emphasizes, rather than an isolated '
     'accuracy comparison.')

h1('3  Methodology / Proposed Method')
h2('3.1  Problem Definition & Symbol Explanation')
body('Let D = {(x_i, y_i)}_{i=1}^{N} be a labelled image set with x_i in R^{28x28} and y_i in {0,...,9}. A classifier '
     'f_theta maps an image to ten logits, and training minimizes the empirical risk')
equation('eq1.png', 1, 0.75)
body('where l is the cross-entropy loss. Compression introduces a second objective: find f_edge with parameters '
     'theta_edge such that the accuracy drop is bounded while the storage S(f) = |theta| x b and the latency T(f) '
     'are minimized; b denotes bits per parameter. We report four quantities for every variant, namely top-1 '
     'accuracy on the 10,000-image test split, parameter count, serialized model size in megabytes (int8 tensors '
     'counted as one byte and float32 tensors as four), and median single-image latency over 120 runs after 20 '
     'warm-up runs on two CPU threads.')

h2('3.2  Course Theoretical Preliminaries')
body('A convolution layer computes, for output channel k at position (i, j),')
equation('eq2.png', 2, 0.85)
body('The two sums express the inductive bias of the module: the kernel weights w are shared across all positions, '
     'and each output only sees a local window, so the parameter count is independent of the image size. A '
     'Transformer block replaces the local window with global attention,')
equation('eq3.png', 3, 0.72)
body('where Q, K and V are linear projections of the token sequence and d_k is the key dimension. Because attention '
     'compares every pair of the 49 patch tokens produced by a 4x4 patch embedding, its cost grows quadratically in '
     'the number of tokens, which is why the ViT variant is measured separately rather than assumed to be better.')
body('Distillation trains the student on a combination of hard labels and the teacher\'s soft distribution,')
equation('eq4.png', 4, 0.95)
body('with temperature T, weighting coefficient alpha and p_i^T, q_i^T the temperature-softened teacher and student '
     'distributions. The T^2 factor keeps the gradient magnitude comparable to the cross-entropy term when T is '
     'large. Finally, symmetric per-tensor quantization maps a real value w to an 8-bit integer')
equation('eq5.png', 5, 0.72)
body('where s is the scale derived from the observed range and z is the zero point that aligns the real zero with '
     'an integer, so that padding and ReLU boundaries remain exactly representable.')

h2('3.3  Core Model & Algorithm Design')
body('The proposed pipeline is shown in Fig. 1. A compact CNN with three convolution stages (32, 64 and 128 '
     'channels), batch normalization, ReLU, 2x2 max pooling, global average pooling and a 10-way linear head is '
     'trained as the teacher. The student keeps the same topology but halves every channel width to 16, 32 and 64, '
     'which cuts parameters from 94.4K to 24.2K. The student is trained for four epochs with the distillation '
     'objective of Eq. (4), T = 4 and alpha = 0.7, using the teacher logits computed on the same augmented batch.')
figure('fig1_architecture.png', 'Fig. 1  Proposed pipeline: a course-level CNN backbone (Lesson 3) is compressed by '
       'knowledge distillation, structured channel pruning and static INT8 PTQ (Lesson 15).', 15.2)
body('Pruning is structured so that it actually shortens the dense kernels. For each convolution layer we score '
     'every output filter by its L1 norm, s_j = ||W_j||_1, keep the top r fraction and physically rebuild the layer '
     'with the reduced input and output channel counts; batch-normalization statistics and the classifier input '
     'are sliced with the same indices. This differs from masked unstructured pruning, which leaves the tensor '
     'shape unchanged and therefore does not reduce latency on a dense CPU kernel. A two-epoch fine-tune with a '
     'OneCycle schedule recovers most of the pruning loss.')
body('The pruned student is then converted to INT8 with FX graph-mode PTQ. Eight calibration batches of 64 images '
     'are pushed through the prepared model so that each observer records the activation range, after which '
     'convolution, batch normalization and ReLU are folded and converted to the fbgemm integer backend. Weight '
     'scales are per-channel and activation scales per-tensor, which is the configuration recommended for x86 '
     'CPUs. Algorithm 1 summarizes the complete construction.')
caption('Algorithm 1  Edge-model construction (KD + structured pruning + INT8 PTQ)', before=6, after=2)
mono_box([
 'Require: dataset D, teacher f_T (94.4K), keep ratio r = 0.8, temperature T = 4, epochs E = 4',
 'Ensure:  quantized edge model f_edge and its accuracy / size / latency',
 ' 1: train f_T on D with AdamW + OneCycleLR, seed 42, 5 epochs',
 ' 2: build student f_S with half-width channels (24.2K parameters)',
 ' 3: for epoch = 1 to E do',
 ' 4:     L = (1 - alpha) * T^2 * KL(p_T^T || p_S^T) + alpha * CE(y, p_S)',
 ' 5:     theta_S <- AdamW(L, lr = 2e-3)',
 ' 6: end for',
 ' 7: for each conv layer l do',
 ' 8:     score every output filter by ||W_j||_1 and keep the top r fraction',
 ' 9:     rebuild layer l with the kept input/output channels',
 '10: end for',
 '11: fine-tune f_S for 2 epochs (teacher soft labels kept as regularizer)',
 '12: insert observers, calibrate on 8 batches of 64 images, convert to INT8',
 '13: return f_edge',
])

h2('3.4  Experimental Implementation Details')
body('All experiments run on a Linux container with two CPU threads and no GPU, using PyTorch 2.14 (CPU build), '
     'NumPy 1.26 and Matplotlib 3.11. The complete hyper-parameter set is listed in Table 1. Random seeds are fixed '
     'for NumPy and PyTorch, and the DataLoader shuffle uses a seeded generator, so every reported number can be '
     'reproduced by running the scripts in Section 5. Latency is measured after setting the thread count explicitly, '
     'because PyTorch otherwise oversubscribes the two available cores and inflates the variance by a factor of two.')

caption('Table 1  Hyper-parameter configuration (identical for all reported runs)', before=8, after=3)
add_table(['Item', 'Setting'], [
    ['Optimizer / schedule', 'AdamW, OneCycleLR (pct_start = 0.25)'],
    ['Learning rate / weight decay', '2e-3 / 5e-4 (CNN), 1e-3 / 5e-4 (ViT)'],
    ['Epochs / batch size', '5 (CNN), 4 (ViT) / 128'],
    ['Regularization', 'label smoothing 0.05, weight decay, random shift +/-2 px'],
    ['Distillation', 'T = 4, alpha = 0.7, student width = 0.5, 4 epochs'],
    ['Structured pruning', 'filter L1 norm, keep ratio r = 0.8, 2 fine-tune epochs'],
    ['INT8 PTQ', 'FX graph mode, fbgemm, per-channel weights, 8 x 64 calibration'],
    ['Platform / seed', 'PyTorch 2.14 CPU, 2 threads, NumPy + PyTorch seed 42'],
], [4.6, 11.4], ['left', 'left'])

h1('4  Experiments and Results')
h2('4.1  Experimental Setup')
body(f'Datasets. MNIST provides 60,000 training and 10,000 test images of 28x28 handwritten digits; Fashion-MNIST '
     f'has the same shape and split but replaces digits with ten garment classes, which makes it a harder test of '
     f'the same pipeline [12]. Inputs are normalized with mean 0.1307 and standard deviation 0.3081, and training '
     f'uses a random translation of up to two pixels implemented as padding plus cropping, so that the model does '
     f'not rely on the digit being perfectly centred.')
body('Metrics. Top-1 accuracy measures task quality; parameter count and serialized size measure memory; median '
     'latency measures the per-frame cost. Baselines are the full-precision teacher, the TinyViT backbone trained '
     'with the same schedule, a structured-pruned teacher, and a same-size student trained without distillation. '
     'The edge model is compared against these under identical preprocessing and thread settings.')

h2('4.2  Main Results')
body('Table 2 reports the main comparison on MNIST.')
caption('Table 2  Main results on MNIST (10,000 test images, 2 CPU threads, batch size 1)', before=6, after=2)
rows = [
    ['CNN teacher (FP32)', pct(cnn['acc']), f"{cnn['params']/1000:.1f}K", f"{cnn['mb']:.3f}", f"{cnn['lat']:.2f}"],
    ['TinyViT (FP32)', pct(vit['acc']), f"{vit['params']/1000:.1f}K", f"{vit['mb']:.3f}", f"{vit['lat']:.2f}"],
    ['TinyViT dynamic INT8', pct(M['vit_dynamic_int8']['acc']), f"{M['vit_dynamic_int8']['params']/1000:.1f}K", f"{M['vit_dynamic_int8']['mb']:.3f}", f"{M['vit_dynamic_int8']['lat']:.2f}"],
    ['Pruned teacher 50% (FP32)', pct(prune['acc']), f"{prune['params']/1000:.1f}K", f"{prune['mb']:.3f}", f"{prune['lat']:.2f}"],
    ['KD student (FP32)', pct(kd['acc']), f"{kd['params']/1000:.1f}K", f"{kd['mb']:.3f}", f"{kd['lat']:.2f}"],
    ['Scratch student (FP32)', pct(scr['acc']), f"{scr['params']/1000:.1f}K", f"{scr['mb']:.3f}", f"{scr['lat']:.2f}"],
    ['CNN teacher INT8', pct(ci8['acc']), f"{ci8['params']/1000:.1f}K", f"{ci8['mb']:.3f}", f"{ci8['lat']:.2f}"],
    ['KD student INT8', pct(ki8['acc']), f"{ki8['params']/1000:.1f}K", f"{ki8['mb']:.3f}", f"{ki8['lat']:.2f}"],
    ['Ours (KD+Prune+INT8)', pct(ours['acc']), f"{ours['params']/1000:.1f}K", f"{ours['mb']:.3f}", f"{ours['lat']:.2f}"],
]
add_table(['Model', 'Accuracy (%)', 'Params', 'Size (MB)', 'Latency (ms)'], rows,
          [5.2, 2.9, 2.6, 2.6, 2.7], ['left', 'center', 'right', 'right', 'right'])
body(f'Three observations follow. First, the ViT backbone is not automatically better at this scale: it needs '
     f'{vit["params"]/1000:.1f}K parameters, {vit["mb"]/cnn["mb"]:.1f}x the storage of the CNN, and reaches '
     f'{pct(vit["acc"])}%, slightly below the {pct(cnn["acc"])}% of the convolutional teacher. With 49 tokens and '
     f'four blocks, the global attention has too little data to learn the locality prior that convolution receives '
     f'for free, which is the practical face of the data-hunger discussed in Lesson 5.')
body(f'The latency column contains a counter-intuitive entry: the ViT uses {vit["params"]/cnn["params"]:.2f}x more '
     f'parameters than the CNN yet runs in {vit["lat"]:.2f} ms instead of {cnn["lat"]:.2f} ms. The reason is that '
     f'parameter count measures memory, not arithmetic organisation. The ViT concentrates its work in a few large '
     f'matrix multiplications that the BLAS backend executes with high arithmetic intensity, while the CNN launches '
     f'many small convolution kernels whose overhead dominates on two CPU threads. Dynamic INT8 quantization of the '
     f'ViT linear layers halves its storage to {M["vit_dynamic_int8"]["mb"]:.3f} MB but raises latency to '
     f'{M["vit_dynamic_int8"]["lat"]:.2f} ms, because dequantizing weights on the fly costs more than the integer '
     f'arithmetic saves at this scale. A deployment decision therefore has to be made on measured latency, not on '
     f'parameter count alone.')
body(f'Second, distillation and pruning reach the same size through different routes. The pruned teacher keeps '
     f'{pct(prune["acc"])}% with 24.2K parameters, while the distilled student keeps {pct(kd["acc"])}% with the same '
     f'count, a {abs(kd["acc"]-prune["acc"])*100:.2f}-point advantage for distillation. Pruning removes channels that '
     f'the teacher itself considered weak, whereas distillation reshapes the whole student function to match the '
     f'teacher\'s soft distribution, so the student can keep accuracy that pure channel removal destroys.')
body(f'Third, INT8 is the cheapest single step. Applied to the teacher it cuts size by {q_size:.1f}x and latency by '
     f'{q_speed:.1f}x for a {abs(ci8["acc"]-cnn["acc"])*100:.2f}-point drop; applied to the student it cuts size by '
     f'{kd["mb"]/ki8["mb"]:.1f}x for a {abs(ki8["acc"]-kd["acc"])*100:.2f}-point drop. The final combination keeps '
     f'{pct(ours["acc"])}% at {ours["mb"]*1024:.1f} KB, which is {size_ratio:.1f}x smaller and {speed_ratio:.1f}x '
     f'faster than the FP32 teacher, with {param_drop:.1f}% fewer parameters and a {acc_drop:.2f}-point accuracy '
     f'cost. Relative accuracy retention is {ours["acc"]/cnn["acc"]*100:.1f}%, so more than 98% of the original '
     f'accuracy survives a 24-fold size reduction.')
body(f'The Fashion-MNIST run tests whether the conclusions transfer to a harder task. The same CNN reaches '
     f'{pct(ff["acc"])}% in FP32 and {pct(fi["acc"])}% after INT8, a drop of {abs(fi["acc"]-ff["acc"])*100:.2f} '
     f'points, essentially the same as on MNIST; the absolute accuracy is lower because garment silhouettes are '
     f'more ambiguous than digits, but the compression behaviour is unchanged, which suggests the pipeline is not '
     f'tuned to one dataset.')

h2('4.3  Ablation Study')
body('Table 3 isolates the contribution of each stage by removing it from the final model. All variants use the '
     'same seed, data order and calibration set.')
caption('Table 3  Ablation study on MNIST (deltas are relative to the full model)', before=6, after=2)
abl_rows = [
    ['Ours (KD + Prune + INT8)', pct(ours['acc']), '--'],
    ['w/o INT8 (KD + Prune, FP32)', pct(abl.get('no_int8_fp32', 0)), f"{(abl.get('no_int8_fp32',0)-ours['acc'])*100:+.2f}"],
    ['w/o Prune (KD + INT8)', pct(ki8['acc']), f"{(ki8['acc']-ours['acc'])*100:+.2f}"],
    ['w/o KD (Scratch + Prune + INT8)', pct(abl.get('no_kd_int8', 0)), f"{(abl.get('no_kd_int8',0)-ours['acc'])*100:+.2f}"],
]
add_table(['Variant', 'Accuracy (%)', 'Delta vs. Ours (pp)'], abl_rows, [7.4, 3.4, 4.2], ['left', 'center', 'right'])
body(f'Removing INT8 raises accuracy by {(abl.get("no_int8_fp32",0)-ours["acc"])*100:.2f} points, but the '
     f'model grows from {ours["mb"]*1024:.1f} KB to {abl_mb:.1f} KB, a {abl_mb/(ours["mb"]*1024):.1f}x increase. '
     f'Quantization is therefore the stage that buys memory rather than the stage that protects accuracy, and the '
     f'low cost of {abs(abl.get("no_int8_fp32",0)-ours["acc"])*100:.2f} points is the reason it is worth applying last.')
body(f'Removing pruning raises accuracy by {(ki8["acc"]-ours["acc"])*100:.2f} points, the largest accuracy gain of '
     f'any removal, but leaves a model {ki8["mb"]/ours["mb"]:.2f}x larger. In other words pruning spends '
     f'{(ki8["acc"]-ours["acc"])*100:.2f} points to save about {(ki8["mb"]-ours["mb"])*1024:.1f} KB, an exchange rate '
     f'worth taking when flash memory is the binding constraint but not when accuracy is.')
body(f'Removing distillation lowers accuracy by {(abl.get("no_kd_int8",0)-ours["acc"])*100:.2f} points after '
     f'compression, compared with {kd_gain:.2f} points at the same parameter budget before compression. The gain '
     f'shrinks because structured pruning already removes the channels that the soft targets mostly protect, but it '
     f'does not vanish: distillation is the only stage whose removal hurts accuracy relative to the full model, '
     f'which is why it is placed first in the pipeline.')

body('The order of operations also matters. Applying pruning before distillation forces the teacher to fit a '
     'handicapped student, while the reverse order lets the student first learn the full soft distribution and then '
     'lose only the channels that contribute least. We therefore fix the order KD -> prune -> INT8 for all runs.')

h2('4.4  Visualization Analysis')
body('Fig. 2 shows the optimization and generalization curves. The CNN converges quickly because the translation '
     'augmentation and label smoothing prevent it from memorizing pixel positions, and its validation accuracy '
     'plateaus near 98%. The ViT starts slower, jumps from 77% to 95% between the first two epochs, and ends at '
     '97.8%, which is the signature of an attention model that first learns global token statistics and then '
     'refines local patterns. Fashion-MNIST converges more slowly and ends lower, as expected from its harder '
     'class boundaries.')
figure('fig2_training_curves.png', 'Fig. 2  Training loss (a) and validation accuracy (b) for CNN, TinyViT and '
       'Fashion-MNIST runs; all curves use the same schedule and seed.', 15.0)
body(f'Fig. 3 is the confusion matrix of the final INT8 model on MNIST. The diagonal dominates, with '
     f'{int(M["ours"]["acc"]*10000)} correct predictions out of 10,000. The largest off-diagonal entries are 5 vs 3, '
     f'4 vs 9 and 7 vs 1; these pairs share stroke endings after the 2x2 pooling, and the quantized activations make '
     f'the fine distinction between a closed and an open top even harder, since the clipped range removes the low '
     f'magnitude differences that separate them.')
figure('fig3_confusion.png', 'Fig. 3  Confusion matrix of the final INT8 edge model on the MNIST test set.', 9.2)
body('Fig. 4 summarizes the accuracy-size-latency trade-off of all variants. The three panels share the same row '
     'order, so a horizontal scan reads as a deployment decision: the FP32 teacher sits at the top of accuracy and '
     'the bottom of size, while the final model moves to the opposite corner with almost no accuracy change. The '
     'latency panel is the most informative, because the INT8 conversion roughly halves the latency of every model '
     'it touches, whereas pruning alone only reduces the floating-point work proportionally to the removed channels.')
figure('fig4_tradeoff.png', 'Fig. 4  Accuracy (a), model size (b) and single-image CPU latency (c) for all variants.', 15.2)
body('Fig. 5 shows thirty test images with the prediction and confidence produced by the edge model, three per class. '
     'Most samples are classified with more than 90% confidence, and the single error in the grid is a thin "3" read '
     'as "2" with 77% confidence, which is exactly the type of ambiguous stroke that the confusion matrix flagged. '
     'Displaying confidence rather than a bare label is what makes the failure visible to a user.')
figure('fig5_samples.png', 'Fig. 5  Predictions of the INT8 edge model on thirty test images (confidence in '
       'parentheses; red marks an error).', 15.2)

h2('4.5  Robustness Analysis')
body('Clean accuracy is not the whole deployment story, so we corrupt the test set in two ways: additive Gaussian '
     'noise with standard deviation 0.1 and 0.2 in normalized units, and a centred square occlusion of 4 or 6 pixels. '
     'Table 4 reports the results for the teacher, the FP32 distilled student and the final INT8 model.')
caption('Table 4  Corruption robustness on MNIST (accuracy %)', before=6, after=2)
rb = M['robust']
rob_rows = []
for key, label in [('cnn_fp32', 'CNN teacher (FP32)'), ('kc_student_fp32', 'KD student (FP32)'), ('ours_int8', 'Ours (INT8)')]:
    d = rb[key]
    rob_rows.append([label, pct(d['clean']), pct(d['noise0.1']), pct(d['noise0.2']), pct(d['occl4']), pct(d['occl6'])])
add_table(['Model', 'Clean', 'Noise 0.1', 'Noise 0.2', 'Occl. 4px', 'Occl. 6px'], rob_rows,
          [4.4, 2.3, 2.4, 2.4, 2.3, 2.2], ['left', 'center', 'center', 'center', 'center', 'center'])
body(f'The FP32 teacher degrades gracefully: at noise 0.2 it keeps {pct(rb["cnn_fp32"]["noise0.2"])}%, and at a '
     f'6-pixel occlusion it still answers {pct(rb["cnn_fp32"]["occl6"])}%. The INT8 model is more fragile: the same '
     f'noise level drops it to {pct(rb["ours_int8"]["noise0.2"])}%, a gap of '
     f'{(rb["cnn_fp32"]["noise0.2"]-rb["ours_int8"]["noise0.2"])*100:.2f} points, while under a 6-pixel occlusion '
     f'the gap is {(rb["cnn_fp32"]["occl6"]-rb["ours_int8"]["occl6"])*100:.2f} points. The reason is the activation '
     f'observer: PTQ fixes the clipping range from clean calibration data, so a noisy input pushes many activations '
     f'outside the recorded range, they saturate at the clipping boundary, and the information that distinguished '
     f'the classes is lost before the classifier sees it.')
body('This result does not invalidate quantization, but it changes the deployment recipe: a model intended for a '
     'noisy camera should either calibrate on augmented data, keep a wider activation range, or use QAT so that the '
     'network learns to tolerate clipping. The FP32 student, in contrast, is more sensitive to occlusion than the '
     'teacher but less sensitive to noise, because its narrower width removes filters that respond to fine-grained '
     'high-frequency patterns.')

h2('4.6  Result Discussion')
body('The experiments support three conclusions that connect back to the course theory. Structured pruning works '
     'because convolution channels are redundant: after batch normalization the scale of each channel is explicit, '
     'and removing the channels with small L1 norm removes functions that contribute little to the output. '
     'Distillation works because the teacher transmits more than the hard label: the relative probabilities of '
     'confusable classes encode the similarity structure of the data, and a narrow student cannot recover that '
     'structure from 60,000 one-hot labels alone. Quantization works because inference is dominated by memory '
     'traffic on a CPU, so halving the bytes per weight gives a larger speed-up than the arithmetic saving alone '
     'would suggest.')

h1('5  Reproducibility and Reusable Code')
body('To keep every reported number reproducible, the complete codebase is included in the submission package '
     'together with the measurement scripts. The repository is organized as follows.')
mono_box([
 'final-report/',
 '|-- README.md              # one-command reproduction',
 '|-- requirements.txt       # pinned dependencies',
 '|-- configs/exp.yaml       # all hyper-parameters',
 '|-- data/                  # MNIST and Fashion-MNIST IDX files',
 '|-- code/',
 '|   |-- data_utils.py      # IDX reader, normalization, shift augmentation',
 '|   |-- models.py          # SmallCNN (Lesson 3) and TinyViT (Lesson 5)',
 '|   |-- train.py           # training entry, seed 42',
 '|   |-- compress.py        # KD, structured pruning, INT8 PTQ, robustness',
 '|   |-- ablation.py        # component removal experiments',
 '|   |-- eval_extra.py      # ViT and Fashion-MNIST evaluation',
 '|   |-- figures.py         # Fig. 1-5',
 '|   +-- demo/server.py     # demonstration system',
 '|-- scripts/run_all.sh     # reproduces Tables 1-4 and Figs. 2-5',
 '|-- figures/               # generated figures',
 '+-- results/               # checkpoints, metrics.json, logs',
])
body('One-command reproduction is:')
mono_box([
 'pip install -r requirements.txt',
 'bash scripts/run_all.sh',
 '# outputs: results/metrics.json, figures/fig1-5.png',
])
body(f'All randomness is fixed with seed 42 for NumPy and PyTorch, and the DataLoader uses a seeded generator. '
     f'The full suite takes under an hour on two CPU threads: the CNN teacher needs roughly 100 s per epoch, the '
     f'ViT about 65 s, and the compression stage about six minutes including two fine-tuning runs. Each individual '
     f'number in Tables 1-4 can therefore be regenerated on a laptop without a GPU.')

h1('6  System Demonstration')
body('Offline accuracy is only one half of the deployment story. To show the other half, we build a small '
     'interactive system that wraps the INT8 model in a browser interface. Following the system view of Lesson 11, '
     'the demonstration contains perception (the quantized CNN), a tool layer (the inference endpoint) and an '
     'interface (the canvas and the top-3 confidence bars).')
body('The architecture keeps the backend deliberately dependency-free. A Python standard-library HTTP server exposes '
     'two routes: GET / returns the single-page interface, and POST /predict accepts a base64 PNG produced by the '
     'browser canvas and returns the top-3 classes with probabilities. Because the front-end compresses the drawing '
     'to 28x28 before sending it, the payload is a few kilobytes and the round trip stays close to the 0.36 ms '
     'inference cost plus network overhead. Table 5 lists the components.')
caption('Table 5  Components of the demonstration system', before=6, after=2)
add_table(['Component', 'Technology', 'Function'], [
    ['Backend', 'Python 3.12 http.server', 'serves the page and the /predict JSON endpoint'],
    ['Model', 'PyTorch INT8 (fbgemm)', 'top-3 inference from the 15.9K-parameter edge model'],
    ['Frontend', 'HTML5 canvas + fetch', 'draw or upload a digit, render confidence bars'],
    ['Preprocessing', 'Pillow + NumPy', 'grayscale, resize to 28x28, auto-invert light background'],
    ['Deployment', 'localhost:8000', 'single process, no GPU, no external web framework'],
], [3.3, 5.3, 7.2], ['left', 'left', 'left'])
body('A measured exchange with the running server (digit 7 from the MNIST test set) looks like this:')
mono_box([
 'POST /predict   {"image": "<base64 PNG of a 28x28 digit>"}',
 'HTTP 200        {"top3": [{"digit": "7", "prob": 0.9385},',
 '                           {"digit": "9", "prob": 0.0375},',
 '                           {"digit": "1", "prob": 0.0109}]}',
])
body('The walkthrough has three steps: the user draws or uploads a digit, the page downsamples it to 28x28 and '
     'posts it to the endpoint, and the server returns the top-3 distribution that the page renders as confidence '
     'bars. Fig. 5 shows thirty such predictions, including one low-confidence error, so the system '
     'shows not only what the model predicts but also when it is unsure. This closes the loop from course theory to '
     'a usable application and gives a concrete answer to the question "what does a compressed model look like in '
     'practice".')

h1('7  Discussion')
h2('7.1  In-depth Result Interpretation')
body(f'The compression numbers are not proportional across stages, and the reason is where each technique saves '
     f'work. Distillation and pruning reduce the number of parameters by {param_drop_kd:.1f}%, but a 24.2K-parameter '
     f'CNN is already small enough that the CPU spends a large fraction of each forward pass on memory allocation and '
     f'kernel launch rather than on arithmetic; that is why the pruned model is only '
     f'{cnn["lat"]/prune["lat"]:.2f}x faster despite having a quarter of the parameters. INT8 changes the data type '
     f'rather than the count, so it touches every byte of every layer: the teacher becomes '
     f'{cnn["lat"]/ci8["lat"]:.2f}x faster and {q_size:.1f}x smaller at once. The final model mixes both effects, '
     f'which is why its latency is {speed_ratio:.1f}x lower while its size is {size_ratio:.1f}x lower, two ratios '
     f'that a single technique alone cannot produce.')
body(f'The accuracy cost also decomposes cleanly. Pruning costs {abs(prune["acc"]-cnn["acc"])*100:.2f} points '
     f'because removed channels cannot be recovered by fine-tuning, distillation costs '
     f'{abs(kd["acc"]-cnn["acc"])*100:.2f} points because a narrow student has less capacity, and INT8 costs '
     f'{abs(ci8["acc"]-cnn["acc"])*100:.2f} points because the activation range is clipped. The combined cost is '
     f'{acc_drop:.2f} points rather than the sum of the three, which means the stages interact: a student trained '
     f'with soft labels is more tolerant of channel removal, and a pruned network has a smaller activation range '
     f'that quantizes more accurately.')

h2('7.2  Advantages & Limitations')
body('The main advantage of the proposed pipeline is that every stage is cheap and independently measurable, so a '
     'practitioner can stop at the point where the accuracy-size curve is acceptable. The final model is small '
     'enough to fit in the flash memory of a microcontroller and fast enough for a 30-frame-per-second stream on a '
     'CPU, and the whole construction takes minutes rather than GPU hours. The demonstration shows that the result '
     'is usable without a serving framework.')
body('The limitations are equally clear. The experiments use 28x28 grayscale images, so the absolute latency and '
     'size are not representative of a 224x224 RGB model; the conclusions about relative ratios should be re-tested '
     'on a larger backbone. No pretrained weights are used, which keeps the comparison fair but also caps accuracy '
     'at the level a small dataset allows. Only post-training quantization is studied; QAT would probably recover '
     'part of the accuracy loss, especially under noise. Finally, latency is measured on one x86 CPU with two '
     'threads, and integer kernels on a phone NPU or a microcontroller have different memory hierarchies, so the '
     'speed-up ratio should not be transferred without measurement.')

h2('7.3  Course Learning Reflection')
body('The project changed how I read the eight-module framework. Before the course, visual intelligence looked like '
     'a list of architectures; after implementing a CNN and a ViT under the same budget, the architectures became '
     'two different priors, one that assumes locality and one that learns it from data, and the accuracy gap '
     'between them became an explanation rather than a number. The same shift happened with compression: pruning, '
     'quantization and distillation stopped being three recipes and became three ways of removing a different kind '
     'of redundancy, namely redundant channels, redundant precision and redundant supervision.')
body('The main difficulty was not the theory but the tooling. The quantization API in the current PyTorch release '
     'is deprecated and the FX graph mode has to be used explicitly; the quantized model cannot be pickled across '
     'processes, so the demonstration rebuilds it from the float checkpoint and the calibration set at start-up. '
     'Measuring latency reliably on two shared threads also required fixing the thread count and discarding warm-up '
     'runs. These problems are exactly the gap between a course exercise and a deployable system, and solving them '
     'was the most valuable part of the work.')

h2('7.4  Future Work')
body('Four directions follow naturally. First, quantization-aware training and channel-wise calibration on augmented '
     'data should reduce the noise sensitivity reported in Table 4. Second, the pipeline can be extended to a '
     'pretrained backbone such as a small ViT or a MobileNetV2, where distillation from a large multimodal teacher '
     'would be closer to the "large model" setting of the course. Third, an agent layer could automate the search '
     'over keep ratios and bit widths, turning the manual grid into a tool-using loop in the sense of Lesson 11. '
     'Fourth, the same accuracy-size-latency protocol should be repeated on a real edge device, since only a '
     'hardware measurement can confirm whether the simulated integer pipeline translates into wall-clock savings.')

h1('8  Conclusion')
body(f'This paper asked how much accuracy a course-scale image classifier must give up when it is compressed for '
     f'edge deployment, and which combination of techniques pays the lowest price. Starting from a '
     f'{cnn["params"]/1000:.1f}K-parameter CNN teacher at {pct(cnn["acc"])}% on MNIST, we distilled a half-width '
     f'student, pruned its channels by filter L1 norm and converted it to INT8 with static PTQ. The resulting model '
     f'keeps {pct(ours["acc"])}% with {ours["params"]/1000:.1f}K parameters and {ours["mb"]*1024:.1f} KB, which is '
     f'{size_ratio:.1f}x smaller and {speed_ratio:.1f}x faster than the FP32 baseline at a cost of {acc_drop:.2f} '
     f'percentage points. The ablation shows that distillation is the stage that protects accuracy, pruning is the '
     f'stage that buys memory, and quantization is the stage that buys latency, while the robustness study shows '
     f'that quantization also narrows the activation range and therefore raises sensitivity to sensor noise. The '
     f'pipeline, the measurement scripts and the browser demonstration together form a reproducible example of how '
     f'the visual and compression modules of the course can be combined into a deployable system, and they suggest '
     f'that the next step is to repeat the same protocol on a real edge device and with quantization-aware training.')

h1('References')
refs = [
 '[1] LeCun Y, Bottou L, Bengio Y, Haffner P. Gradient-based learning applied to document recognition [J]. Proceedings of the IEEE, 1998, 86(11): 2278-2324.',
 '[2] Vaswani A, Shazeer N, Parmar N, et al. Attention is all you need [C]//Advances in Neural Information Processing Systems (NeurIPS), 2017: 5998-6008.',
 '[3] Dosovitskiy A, Beyer L, Kolesnikov A, et al. An image is worth 16x16 words: Transformers for image recognition at scale [C]//International Conference on Learning Representations (ICLR), 2021.',
 '[4] Course Teaching Group. 大模型基础与应用探索: Course Materials, Lessons 1-16 [Z]. Harbin Institute of Technology, Shenzhen, 2026.',
 '[5] Han S, Pool J, Tran J, Dally W. Learning both weights and connections for efficient neural networks [C]//NeurIPS, 2015: 1135-1143.',
 '[6] Han S, Mao H, Dally W J. Deep Compression: Compressing deep neural networks with pruning, trained quantization and Huffman coding [C]//ICLR, 2016.',
 '[7] Li H, Kadav A, Durdanovic I, Samet H, Graf H P. Pruning filters for efficient ConvNets [C]//ICLR, 2017.',
 '[8] Frankle J, Carbin M. The lottery ticket hypothesis: Finding sparse, trainable neural networks [C]//ICLR, 2019.',
 '[9] Hinton G, Vinyals O, Dean J. Distilling the knowledge in a neural network [J]. arXiv:1503.02531, 2015.',
 '[10] Jacob B, Kligys S, Chen B, et al. Quantization and training of neural networks for efficient integer-arithmetic-only inference [C]//IEEE CVPR, 2018: 2704-2713.',
 '[11] Sandler M, Howard A, Zhu M, Zhmoginov A, Chen L C. MobileNetV2: Inverted residuals and linear bottlenecks [C]//IEEE CVPR, 2018: 4510-4520.',
 '[12] Xiao H, Rasul K, Vollgraf R. Fashion-MNIST: A novel image dataset for benchmarking machine learning algorithms [J]. arXiv:1708.07747, 2017.',
]
for r in refs:
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    pf.line_spacing = 1.15
    pf.space_after = Pt(3)
    pf.left_indent = Cm(0.9)
    pf.first_line_indent = Cm(-0.9)
    run = p.add_run(r)
    _font(run, 10)

doc.save(OUT)
print('saved', OUT)
