"""轻量化实验：结构化剪枝、知识蒸馏、静态 INT8 量化、鲁棒性评测。"""
import copy, json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_utils import get_loaders
from models import SmallCNN, count_params

torch.set_num_threads(2)


def model_bytes(model):
    total = 0
    for p in model.state_dict().values():
        if isinstance(p, torch.Tensor):
            n = p.numel()
            if p.dtype in (torch.int8, torch.uint8, torch.quint8, torch.qint8):
                total += n
            elif p.dtype == torch.float32:
                total += n * 4
            elif p.dtype == torch.float16:
                total += n * 2
            else:
                total += n * 4
    return total


@torch.no_grad()
def evaluate(model, loader, noise=0.0, occlusion=0, seed=0):
    model.eval()
    correct = total = 0
    g = torch.Generator().manual_seed(seed)
    for x, y in loader:
        if noise > 0:
            x = x + noise * torch.randn(x.shape, generator=g)
        if occlusion > 0:
            k = occlusion
            x = x.clone()
            x[:, :, 14 - k:14 + k, 14 - k:14 + k] = 0
        logits = model(x)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.numel()
    return correct / total


@torch.no_grad()
def predict_logits(model, loader):
    model.eval()
    outs, ys = [], []
    for x, y in loader:
        outs.append(model(x))
        ys.append(y)
    return torch.cat(outs), torch.cat(ys)


@torch.no_grad()
def latency_ms(model, n=120, warmup=20):
    model.eval()
    x = torch.randn(1, 1, 28, 28)
    for _ in range(warmup):
        model(x)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        model(x)
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ts))


def fine_tune(model, tr, epochs=2, lr=1e-3, teacher=None, T=4.0, alpha=0.7):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * len(tr))
    ce = nn.CrossEntropyLoss()
    for ep in range(epochs):
        model.train()
        for x, y in tr:
            opt.zero_grad()
            logits = model(x)
            if teacher is not None:
                with torch.no_grad():
                    tlog = teacher(x)
                loss = alpha * ce(logits, y) + (1 - alpha) * F.kl_div(
                    F.log_softmax(logits / T, 1), F.softmax(tlog / T, 1), reduction='batchmean') * (T * T)
            else:
                loss = ce(logits, y)
            loss.backward()
            opt.step(); sched.step()
    return model


def structured_prune(model, keep=0.5, teacher_for_ft=None):
    """按卷积核 L1 范数裁剪输出通道，物理重建网络。"""
    sd = model.state_dict()
    widths = [model.features[0].out_channels, model.features[3].out_channels, model.features[7].out_channels]
    w1, w2, w3 = sd['features.0.weight'], sd['features.3.weight'], sd['features.7.weight']
    k1 = int(round(widths[0] * keep)); k2 = int(round(widths[1] * keep)); k3 = int(round(widths[2] * keep))
    idx1 = torch.topk(w1.abs().sum((1, 2, 3)), k1).indices.sort().values
    idx2 = torch.topk(w2.abs().sum((1, 2, 3)), k2).indices.sort().values
    idx3 = torch.topk(w3.abs().sum((1, 2, 3)), k3).indices.sort().values

    new = SmallCNN(widths=(k1, k2, k3))
    nsd = new.state_dict()
    # conv1 / bn1
    nsd['features.0.weight'] = w1[idx1].clone(); nsd['features.0.bias'] = sd['features.0.bias'][idx1].clone()
    for k in ['weight', 'bias', 'running_mean', 'running_var']:
        nsd[f'features.1.{k}'] = sd[f'features.1.{k}'][idx1].clone()
    # conv2
    nsd['features.3.weight'] = w2[idx2][:, idx1].clone(); nsd['features.3.bias'] = sd['features.3.bias'][idx2].clone()
    for k in ['weight', 'bias', 'running_mean', 'running_var']:
        nsd[f'features.4.{k}'] = sd[f'features.4.{k}'][idx2].clone()
    # conv3
    nsd['features.7.weight'] = w3[idx3][:, idx2].clone(); nsd['features.7.bias'] = sd['features.7.bias'][idx3].clone()
    for k in ['weight', 'bias', 'running_mean', 'running_var']:
        nsd[f'features.8.{k}'] = sd[f'features.8.{k}'][idx3].clone()
    nsd['classifier.weight'] = sd['classifier.weight'][:, idx3].clone()
    nsd['classifier.bias'] = sd['classifier.bias'].clone()
    new.load_state_dict(nsd)
    return new


def quantize_static(model, calib_loader):
    """FX 静态 INT8 PTQ（fbgemm），校准 8 个 batch。"""
    from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
    from torch.ao.quantization import get_default_qconfig_mapping
    m = copy.deepcopy(model).eval()
    ex = (torch.randn(1, 1, 28, 28),)
    prepared = prepare_fx(m, get_default_qconfig_mapping('fbgemm'), ex)
    with torch.no_grad():
        for i, (x, _) in enumerate(calib_loader):
            prepared(x)
            if i >= 7:
                break
    return convert_fx(prepared)


def quantize_dynamic(model):
    return torch.ao.quantization.quantize_dynamic(copy.deepcopy(model).eval(), {nn.Linear}, dtype=torch.qint8)


def run():
    out = {}
    tr, te, tr_ds, _ = get_loaders('../data', '', batch_size=128)
    calib = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*tr_ds), batch_size=64)
    teacher = SmallCNN(); teacher.load_state_dict(torch.load('../results/cnn_mnist.pt')); teacher.eval()

    rec = {}
    rec['cnn_fp32'] = dict(acc=evaluate(teacher, te), params=count_params(teacher),
                           mb=model_bytes(teacher) / 2**20, lat=latency_ms(teacher))
    print('cnn_fp32', rec['cnn_fp32'], flush=True)

    # 1) 结构化剪枝 50%
    pruned = structured_prune(teacher, keep=0.5)
    pruned = fine_tune(pruned, tr, epochs=2, lr=1.5e-3)
    rec['prune50'] = dict(acc=evaluate(pruned, te), params=count_params(pruned),
                          mb=model_bytes(pruned) / 2**20, lat=latency_ms(pruned))
    print('prune50', rec['prune50'], flush=True); torch.save(pruned.state_dict(), '../results/prune50.pt')

    # 2) 知识蒸馏：宽度减半学生
    student = SmallCNN(width=0.5)
    student = fine_tune(student, tr, epochs=4, lr=2e-3, teacher=teacher, T=4.0, alpha=0.7)
    rec['kd_student'] = dict(acc=evaluate(student, te), params=count_params(student),
                             mb=model_bytes(student) / 2**20, lat=latency_ms(student))
    print('kd_student', rec['kd_student'], flush=True); torch.save(student.state_dict(), '../results/kd_student.pt')

    # 3) 静态 INT8 量化（教师 / 学生）
    q_teacher = quantize_static(teacher, calib)
    rec['cnn_int8'] = dict(acc=evaluate(q_teacher, te), params=count_params(teacher),
                           mb=model_bytes(q_teacher) / 2**20, lat=latency_ms(q_teacher))
    print('cnn_int8', rec['cnn_int8'], flush=True)
    torch.save(q_teacher, '../results/cnn_int8.pt')

    q_student = quantize_static(student, calib)
    rec['kd_int8'] = dict(acc=evaluate(q_student, te), params=count_params(student),
                          mb=model_bytes(q_student) / 2**20, lat=latency_ms(q_student))
    print('kd_int8', rec['kd_int8'], flush=True); torch.save(q_student, '../results/kd_int8.pt')

    # 4) 组合：蒸馏 + 剪枝 20% + INT8
    combo = structured_prune(student, keep=0.8)
    combo = fine_tune(combo, tr, epochs=2, lr=1e-3, teacher=teacher)
    q_combo = quantize_static(combo, calib)
    rec['ours'] = dict(acc=evaluate(q_combo, te), params=count_params(combo),
                       mb=model_bytes(q_combo) / 2**20, lat=latency_ms(q_combo))
    print('ours', rec['ours'], flush=True); torch.save(q_combo, '../results/ours_int8.pt')

    # 5) 鲁棒性
    rob = {}
    for name, m in [('cnn_fp32', teacher), ('kc_student_fp32', student), ('ours_int8', q_combo)]:
        rob[name] = {
            'clean': evaluate(m, te),
            'noise0.1': evaluate(m, te, noise=0.1),
            'noise0.2': evaluate(m, te, noise=0.2),
            'occl4': evaluate(m, te, occlusion=4),
            'occl6': evaluate(m, te, occlusion=6),
        }
        print(name, rob[name], flush=True)
    rec['robust'] = rob

    # 6) 混淆矩阵与预测样例
    logits, ys = predict_logits(q_combo, te)
    pred = logits.argmax(1)
    cm = torch.zeros(10, 10, dtype=torch.int64)
    for t, p in zip(ys.tolist(), pred.tolist()):
        cm[t, p] += 1
    torch.save({'cm': cm, 'pred': pred, 'label': ys, 'logits': logits}, '../results/final_preds.pt')

    with open('../results/metrics.json', 'w') as f:
        json.dump(rec, f, indent=2)


if __name__ == '__main__':
    run()
