"""消融：分别去掉 INT8 与 KD，量化各自贡献。"""
import json
import torch
from data_utils import get_loaders
from models import SmallCNN
from compress import structured_prune, fine_tune, quantize_static, evaluate

torch.set_num_threads(2)


def main():
    tr, te, tr_ds, _ = get_loaders('../data', '', batch_size=128)
    calib = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*tr_ds), batch_size=64)
    teacher = SmallCNN(); teacher.load_state_dict(torch.load('../results/cnn_mnist.pt')); teacher.eval()
    kd = SmallCNN(width=0.5); kd.load_state_dict(torch.load('../results/kd_student.pt')); kd.eval()
    scratch = SmallCNN(width=0.5); scratch.load_state_dict(torch.load('../results/cnn_small_mnist.pt')); scratch.eval()

    # w/o INT8：KD + Prune，保持 FP32
    a = fine_tune(structured_prune(kd, 0.8), tr, epochs=2, lr=1e-3, teacher=teacher)
    acc_a = evaluate(a, te)
    # w/o KD：从零学生 + Prune + INT8
    b = fine_tune(structured_prune(scratch, 0.8), tr, epochs=2, lr=1e-3, teacher=None)
    bq = quantize_static(b, calib)
    acc_b = evaluate(bq, te)
    print('w/o INT8', acc_a, 'w/o KD', acc_b, flush=True)

    m = json.load(open('../results/metrics.json'))
    m['ablation'] = {'no_int8_fp32': acc_a, 'no_kd_int8': acc_b}
    json.dump(m, open('../results/metrics.json', 'w'), indent=2)


if __name__ == '__main__':
    main()
