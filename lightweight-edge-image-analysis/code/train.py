"""训练入口：python3 train.py --model cnn --data mnist --epochs 6"""
import argparse, json, os, time, copy
import numpy as np
import torch
import torch.nn as nn
from data_utils import get_loaders, shift_augment
from models import SmallCNN, TinyViT, count_params

torch.set_num_threads(2)


def set_seed(s=42):
    np.random.seed(s); torch.manual_seed(s)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    crit = nn.CrossEntropyLoss(reduction='sum')
    for x, y in loader:
        logits = model(x)
        loss_sum += crit(logits, y).item()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.numel()
    return correct / total, loss_sum / total


def train(model, tr, te, epochs, lr, tag, out_dir, augment=True):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * len(tr), pct_start=0.25)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    history = []
    best_acc, best_state = 0.0, None
    for ep in range(1, epochs + 1):
        model.train(); t0 = time.time(); ls = 0.0; n = 0
        for x, y in tr:
            if augment:
                x = shift_augment(x)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step(); sched.step()
            ls += loss.item() * y.numel(); n += y.numel()
        acc, val_loss = evaluate(model, te)
        history.append({'epoch': ep, 'train_loss': ls / n, 'val_loss': val_loss, 'val_acc': acc,
                        'sec': round(time.time() - t0, 1)})
        print(f'[{tag}] ep{ep} train_loss={ls/n:.4f} val_acc={acc:.4f} ({time.time()-t0:.1f}s)', flush=True)
        if acc > best_acc:
            best_acc, best_state = acc, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), os.path.join(out_dir, f'{tag}.pt'))
    with open(os.path.join(out_dir, f'{tag}_history.json'), 'w') as f:
        json.dump({'tag': tag, 'params': count_params(model), 'best_acc': best_acc, 'history': history}, f, indent=2)
    return best_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='cnn', choices=['cnn', 'vit', 'cnn_small'])
    ap.add_argument('--data', default='mnist', choices=['mnist', 'fashion'])
    ap.add_argument('--epochs', type=int, default=6)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--width', type=float, default=1.0)
    ap.add_argument('--out', default='../results')
    ap.add_argument('--limit', type=int, default=None)
    args = ap.parse_args()
    set_seed(42)
    root = '../data' if args.data == 'mnist' else '../data/fashion'
    prefix = '' if args.data == 'mnist' else 'f'
    tr, te, _, _ = get_loaders(root, prefix, batch_size=128, train_limit=args.limit)
    model = {'cnn': SmallCNN, 'vit': TinyViT, 'cnn_small': lambda: SmallCNN(width=0.5)}[args.model]()
    tag = f'{args.model}_{args.data}' if args.limit is None else f'tmp_{args.model}'
    if args.limit is not None:
        tag += f'_l{args.limit}'
    print(tag, 'params', count_params(model))
    acc = train(model, tr, te, args.epochs, args.lr, tag, args.out)
    print(tag, 'best', acc)


if __name__ == '__main__':
    main()
