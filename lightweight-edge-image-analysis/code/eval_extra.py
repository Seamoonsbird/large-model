"""补充评测：ViT 骨干、Fashion-MNIST 泛化、蒸馏消融。"""
import json, copy
import torch
import torch.nn as nn
from data_utils import get_loaders
from models import SmallCNN, TinyViT, count_params
from compress import evaluate, latency_ms, model_bytes, quantize_static, quantize_dynamic

torch.set_num_threads(2)


def main():
    rec = json.load(open('../results/metrics.json'))

    # 1) ViT / MNIST
    tr, te, tr_ds, _ = get_loaders('../data', '', batch_size=128)
    calib = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*tr_ds), batch_size=64)
    vit = TinyViT(); vit.load_state_dict(torch.load('../results/vit_mnist.pt')); vit.eval()
    rec['vit_fp32'] = dict(acc=evaluate(vit, te), params=count_params(vit),
                           mb=model_bytes(vit) / 2**20, lat=latency_ms(vit))
    q_vit = quantize_dynamic(vit)
    rec['vit_dynamic_int8'] = dict(acc=evaluate(q_vit, te), params=count_params(vit),
                                   mb=model_bytes(q_vit) / 2**20, lat=latency_ms(q_vit))
    print('vit', rec['vit_fp32'], rec['vit_dynamic_int8'], flush=True)

    # 2) 蒸馏消融：同结构学生从零训练
    scratch = SmallCNN(width=0.5)
    scratch.load_state_dict(torch.load('../results/cnn_small_mnist.pt')); scratch.eval()
    rec['scratch_student'] = dict(acc=evaluate(scratch, te), params=count_params(scratch),
                                  mb=model_bytes(scratch) / 2**20, lat=latency_ms(scratch))
    print('scratch', rec['scratch_student'], flush=True)

    # 3) Fashion-MNIST 泛化
    trf, tef, trf_ds, _ = get_loaders('../data/fashion', 'f', batch_size=128)
    calibf = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*trf_ds), batch_size=64)
    cnnf = SmallCNN(); cnnf.load_state_dict(torch.load('../results/cnn_fashion.pt')); cnnf.eval()
    rec['cnn_fashion_fp32'] = dict(acc=evaluate(cnnf, tef), params=count_params(cnnf),
                                   mb=model_bytes(cnnf) / 2**20, lat=latency_ms(cnnf))
    qf = quantize_static(cnnf, calibf)
    rec['cnn_fashion_int8'] = dict(acc=evaluate(qf, tef), params=count_params(cnnf),
                                   mb=model_bytes(qf) / 2**20, lat=latency_ms(qf))
    print('fashion', rec['cnn_fashion_fp32'], rec['cnn_fashion_int8'], flush=True)

    with open('../results/metrics.json', 'w') as f:
        json.dump(rec, f, indent=2)


if __name__ == '__main__':
    main()
