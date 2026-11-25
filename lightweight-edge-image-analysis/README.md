# Lightweight Image Analysis for Edge Deployment

Course final project for **大模型基础与应用探索** (HITSZ, 导师：曾国坤). This package contains the
complete code, the measured checkpoints/logs, the generated figures and the final report.

Pipeline: train CNN / TinyViT on MNIST and Fashion-MNIST → knowledge distillation → structured
channel pruning → static INT8 post-training quantization → single-image latency / size / robustness
measurement → browser demo.

## Package layout

```
lightweight-edge-image-analysis/
├── code/
│   ├── data_utils.py       # IDX reader, normalization, shift augmentation
│   ├── models.py           # SmallCNN (Lesson 3), TinyViT (Lesson 5)
│   ├── train.py            # training entry (AdamW + OneCycleLR, seed 42)
│   ├── compress.py         # KD, structured pruning, INT8 PTQ, robustness
│   ├── ablation.py         # component-removal experiments (Table 3)
│   ├── eval_extra.py       # ViT dynamic INT8, scratch student, Fashion-MNIST
│   ├── export_edge.py      # re-export the edge model (float + TorchScript INT8)
│   ├── finalize_ours.py    # re-measure the exported model, refresh figure data
│   ├── figures.py          # Fig. 1-5
│   ├── equations.py        # Eq. (1)-(5) rendered as PNG
│   ├── build_report.py     # generates report/*.docx from results/metrics.json
│   └── demo/server.py      # stdlib HTTP demo (canvas → /predict → top-3)
├── configs/exp.yaml        # all hyper-parameters
├── scripts/
│   ├── run_all.sh          # one-command reproduction
│   └── download_data.sh    # fetch MNIST / Fashion-MNIST IDX files
├── data/                   # empty, see data/README.md
├── figures/                # fig1-5 + eq1-5 (PNG, generated)
├── results/                # checkpoints, metrics.json, training logs
└── report/                 # Final_Report_Lightweight_Image_Analysis.docx (final deliverable)
```

## Quick start

```bash
pip install -r requirements.txt      # torch 2.14 CPU, numpy, matplotlib, Pillow
bash scripts/download_data.sh        # ~41 MB, MNIST + Fashion-MNIST
bash scripts/run_all.sh              # ~45 min on 2 CPU threads, reproduces tables and figures
```

All randomness is fixed (NumPy + PyTorch seed 42, seeded DataLoader generator); latency is measured
after pinning `torch.set_num_threads(2)` and discarding warm-up runs.

## Key measured results (MNIST, 2 CPU threads)

| Model | Acc (%) | Params | Size (MB) | Latency (ms) |
|---|---|---|---|---|
| CNN teacher (FP32) | 97.99 | 94.4K | 0.362 | 1.31 |
| TinyViT (FP32) | 97.78 | 139.0K | 0.530 | 1.11 |
| KD student (FP32) | 97.63 | 24.2K | 0.093 | 0.93 |
| Ours (KD + prune 0.8 + INT8) | 96.72 | 15.9K | 0.015 | 0.36 |

24.5× smaller and 3.7× faster than the FP32 teacher, at a cost of 1.27 accuracy points.
Ablation (Table 3): w/o INT8 96.93, w/o Prune 97.46, w/o KD 96.39. Robustness (Table 4) shows the
INT8 model is markedly more noise-sensitive (64.31% vs 90.36% at noise σ=0.2).

## Demo

```bash
python3 code/demo/server.py       # http://localhost:8000
```

`GET /` serves the canvas page; `POST /predict` takes `{"image": "<base64 PNG 28x28>"}` and returns
top-3 classes with probabilities. The server loads `results/edge_int8_scripted.pt` (TorchScript).

## Reproducing the report

`code/build_report.py` fills the course template with the numbers in `results/metrics.json`:

```bash
python3 code/build_report.py      # template path is set at the top of the script
```

## Notes / known pitfalls

- The FX-quantized `GraphModule` cannot be pickled reliably in the current PyTorch build
  (`'_modules'` AttributeError); use `results/edge_int8_scripted.pt` (TorchScript) instead. That is
  also why `results/ours_int8.pt` is intentionally not shipped.
- Use `torch.ao.quantization.quantize_fx` with `get_default_qconfig_mapping('fbgemm')`; the legacy
  eager API is deprecated in this build.
- Pruning is physical channel removal (layers are rebuilt), not masking, so the reported latency
  reduction is real rather than theoretical.
