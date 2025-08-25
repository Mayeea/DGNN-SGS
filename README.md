

## 环境配置

### 硬件环境
- **GPU**: RTX 4090 (24GB) * 1
- **CPU**: 16 vCPU Intel(R) Xeon(R) Platinum 8358P CPU @ 2.60GHz
- **内存**: 120GB

### 软件环境
- **PyTorch**: 2.1.0
- **Python**: 3.10 (ubuntu22.04)
- **CUDA**: 12.1

### 依赖安装

```bash
pip install torch==2.1.0
pip install torch-geometric
pip install torch-sparse
pip install torch-scatter
pip install scikit-learn
pip install numpy
pip install scipy
pip install networkx
```

## 数据集

本项目支持以下数据集：

### 引用网络数据集
- **Citeseer**: https://github.com/tkipf/pygcn
- **PubMed**: https://github.com/mengzaiqiao/CAN

### 社交网络数据集
- **UAI2010**: http://linqs.umiacs.umd.edu/projects//projects/lbc/index.html

### 学术网络数据集
- **ACM**: https://github.com/Jhy1993/HAN

### 博客网络数据集
- **BlogCatalog**: https://github.com/mengzaiqiao/CAN

### 图片标注数据集
- **Flickr**: https://github.com/mengzaiqiao/CAN

### 蛋白质网络数据集
- **CoraFull**: https://github.com/abojchevski/graph2gauss/



## 使用方法

### 2. 配置设置

在 `config/` 目录下创建相应的配置文件，例如 `60citeseer.ini`：

```ini
[Model_Setup]
epochs = 200
lr = 0.01
weight_decay = 5e-4
k = 10
nhid1 = 64
nhid2 = 32
dropout = 0.5
beta = 0.1
theta = 0.5
no_cuda = False
no_seed = False
seed = 42

[Data_Setting]
n = 3327
fdim = 3703
class_num = 6
structgraph_path = data/citeseer/citeseer_struct.txt
featuregraph_path = data/citeseer/citeseer_feature_
label_path = data/citeseer/citeseer_label.txt
feature_path = data/citeseer/citeseer_feature.txt
test_path = data/citeseer/citeseer_test.txt
train_path = data/citeseer/citeseer_train.txt
