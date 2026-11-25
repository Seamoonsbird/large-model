"""以导出的 edge_float.pt 为准，重测 ours 的全部指标并更新图表数据。"""
import json
import torch
from data_utils import get_loaders
from models import SmallCNN
from compress import quantize_static, evaluate, latency_ms, model_bytes, predict_logits

torch.set_num_threads(2)

tr, te, tr_ds, _ = get_loaders('../data', '', batch_size=128)
calib = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*tr_ds), batch_size=64)
edge = SmallCNN(widths=(13, 26, 51))
edge.load_state_dict(torch.load('../results/edge_float.pt')); edge.eval()
q = quantize_static(edge, calib)

acc = evaluate(q, te)
rec = {'acc': acc, 'params': sum(p.numel() for p in edge.parameters()),
       'mb': model_bytes(q) / 2**20, 'lat': latency_ms(q)}
print('ours', rec, flush=True)

rob = {'clean': acc, 'noise0.1': evaluate(q, te, noise=0.1), 'noise0.2': evaluate(q, te, noise=0.2),
       'occl4': evaluate(q, te, occlusion=4), 'occl6': evaluate(q, te, occlusion=6)}
print('robust', rob, flush=True)

logits, ys = predict_logits(q, te)
pred = logits.argmax(1)
cm = torch.zeros(10, 10, dtype=torch.int64)
for t, p in zip(ys.tolist(), pred.tolist()):
    cm[t, p] += 1
torch.save({'cm': cm, 'pred': pred, 'label': ys, 'logits': logits}, '../results/final_preds.pt')

m = json.load(open('../results/metrics.json'))
m['ours'] = rec
m['robust']['ours_int8'] = rob
json.dump(m, open('../results/metrics.json', 'w'), indent=2)
print('metrics updated')
