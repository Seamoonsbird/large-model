"""导出可加载的端侧浮点/INT8 模型（与 compress.py 的 ours 配置一致）。"""
import torch
from data_utils import get_loaders
from models import SmallCNN
from compress import structured_prune, fine_tune, quantize_static, evaluate

torch.set_num_threads(2)
torch.manual_seed(42)

tr, te, tr_ds, _ = get_loaders('../data', '', batch_size=128)
calib = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*tr_ds), batch_size=64)
teacher = SmallCNN(); teacher.load_state_dict(torch.load('../results/cnn_mnist.pt')); teacher.eval()
kd = SmallCNN(width=0.5); kd.load_state_dict(torch.load('../results/kd_student.pt')); kd.eval()

edge = fine_tune(structured_prune(kd, 0.8), tr, epochs=2, lr=1e-3, teacher=teacher)
torch.save(edge.state_dict(), '../results/edge_float.pt')
q = quantize_static(edge, calib)
acc = evaluate(q, te)
print('edge float params', sum(p.numel() for p in edge.parameters()), 'int8 acc', acc, flush=True)

try:
    sm = torch.jit.script(q)
    torch.jit.save(sm, '../results/edge_int8_scripted.pt')
    loaded = torch.jit.load('../results/edge_int8_scripted.pt')
    print('scripted reload acc', evaluate(loaded, te), flush=True)
except Exception as e:
    print('script failed:', repr(e)[:300], flush=True)
