"""IDX 数据读取（MNIST / Fashion-MNIST），不依赖 torchvision。"""
import gzip
import os
import numpy as np
import torch


def _read_idx(path):
    with gzip.open(path, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        ndim = magic & 0xFF
        dims = [int.from_bytes(f.read(4), 'big') for _ in range(ndim)]
        buf = f.read()
    data = np.frombuffer(buf, dtype=np.uint8)
    if ndim == 3:
        return data.reshape(dims[0], dims[1], dims[2])
    return data.reshape(dims[0])


def load_idx_dataset(root, prefix=''):
    files = {
        'train_x': os.path.join(root, prefix + 'train-images-idx3-ubyte.gz'),
        'train_y': os.path.join(root, prefix + 'train-labels-idx1-ubyte.gz'),
        'test_x': os.path.join(root, prefix + 't10k-images-idx3-ubyte.gz'),
        'test_y': os.path.join(root, prefix + 't10k-labels-idx1-ubyte.gz'),
    }
    x_tr = _read_idx(files['train_x'])
    y_tr = _read_idx(files['train_y'])
    x_te = _read_idx(files['test_x'])
    y_te = _read_idx(files['test_y'])
    return (x_tr, y_tr), (x_te, y_te)


def to_tensor(x, mean=0.1307, std=0.3081):
    t = torch.from_numpy(x.astype(np.float32) / 255.0).unsqueeze(1)
    return (t - mean) / std


def get_loaders(root, prefix='', batch_size=128, seed=42, augment=True, train_limit=None):
    (x_tr, y_tr), (x_te, y_te) = load_idx_dataset(root, prefix)
    if train_limit is not None:
        x_tr, y_tr = x_tr[:train_limit], y_tr[:train_limit]
    xt = to_tensor(x_tr)
    yt = torch.from_numpy(y_tr.astype(np.int64))
    xe = to_tensor(x_te)
    ye = torch.from_numpy(y_te.astype(np.int64))
    ds = torch.utils.data.TensorDataset(xt, yt)
    g = torch.Generator().manual_seed(seed)
    tr_loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, generator=g, num_workers=0)
    te_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(xe, ye),
                                            batch_size=256, shuffle=False, num_workers=0)
    return tr_loader, te_loader, (xt, yt), (xe, ye)


def shift_augment(x):
    """随机平移增广：pad 2 + 随机裁剪回 28x28（批级偏移，向量化实现）。"""
    b = x.shape[0]
    padded = torch.zeros(b, 1, 32, 32)
    padded[:, :, 2:30, 2:30] = x
    dx, dy = torch.randint(0, 5, (2,)).tolist()
    return padded[:, :, dy:dy + 28, dx:dx + 28]
