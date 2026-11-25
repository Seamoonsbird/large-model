# data/

原始 IDX 文件不随包分发（约 41 MB）。运行 `bash scripts/download_data.sh` 自动下载：

- MNIST：https://storage.googleapis.com/cvdf-datasets/mnist
- Fashion-MNIST：https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion

下载完成后本目录应包含 8 个 `.gz` 文件：

```
train-images-idx3-ubyte.gz        train-labels-idx1-ubyte.gz
t10k-images-idx3-ubyte.gz         t10k-labels-idx1-ubyte.gz
fashion-train-images-idx3-ubyte.gz   fashion-train-labels-idx1-ubyte.gz
fashion-t10k-images-idx3-ubyte.gz    fashion-t10k-labels-idx1-ubyte.gz
```
