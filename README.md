# 基于卷积神经网络的MNIST数据分类系统

## 📋 项目简介

本项目实现了一个完整的基于卷积神经网络的MNIST手写数字分类系统，满足课设的所有基础和创新要求。

### 🎯 核心特性

| 特性 | 说明 |
|------|------|
| **基础CNN实验** | 支持1/2/3层卷积、3×3/5×5/7×7卷积核的对比实验 |
| **轻量化模型** | UltraLightCNN 仅 **32,934参数**，准确率 **>99%** |
| **注意力机制** | SENet-CNN 通道注意力增强特征表达 |
| **空洞卷积** | DilatedCNN 扩大感受野不增参数 |
| **深度可分离卷积** | DepthwiseCNN 参数减少75% |
| **Web演示系统** | Flask后端 + HTML5画板，实时手写识别 |
| **ONNX导出** | 支持部署到移动端和边缘设备 |

### ✅ 满足的课设要求

- ✅ **要求A**: 训练网络并记录每次迭代的损失值
- ✅ **要求B**: 改变卷积层和池化层数，观察分类准确率
- ✅ **要求C**: 改变卷积核大小，观察分类准确率
- ✅ **要求D**: 分析改进的卷积神经网络数据分类的有效性
- ✅ **要求E**: 设计实现基于卷积神经网络的数据分类系统
- ✅ **创新1**: 非标准卷积与注意力机制（SENet/空洞/深度可分离）
- ✅ **创新2**: 轻量化与移动端部署（UltraLightCNN < 50k参数 + ONNX + Web演示）

---

## 🚀 快速开始

### 1. 环境准备

```bash
pip install -r requirements.txt
```

### 2. 快速测试（2-3分钟）

```bash
python quick_test.py
```

### 3. 运行基础实验（30-60分钟）

```bash
python experiments.py
```

### 4. 运行创新实验（注意力机制等）

```bash
python advanced_experiments.py
```

### 5. 训练轻量化模型并启动Web演示

```bash
# 训练UltraLightCNN模型（25 epochs，自动保存最佳模型并导出ONNX）
python train_lightweight.py

# 启动Web手写数字识别演示
python app.py
```

---

## 🏆 轻量化模型 UltraLightCNN

### 架构设计

```
输入 (1, 28, 28)
    ↓
Block1: Conv(1→12) + Conv(12→12) + MaxPool  → 14×14
Block2: Conv(12→24) + Conv(24→24) + MaxPool  → 7×7
Block3: Conv(24→40) + Conv(40→40)           → 7×7
    ↓
GAP (全局平均池化) → Dropout → FC(40→10)
    ↓
输出 (10个类别)
```

### 参数量计算

| 模块 | 层 | 参数量 |
|------|-----|--------|
| Block1 | Conv(1→12) + BN + Conv(12→12) + BN + MaxPool | 1,452 |
| Block2 | Conv(12→24) + BN + Conv(24→24) + BN + MaxPool | 7,872 |
| Block3 | Conv(24→40) + BN + Conv(40→40) + BN | 23,200 |
| Classifier | Dropout + Linear(40→10) | 410 |
| **总计** | | **32,934 ✅** |

### 训练策略

| 策略 | 参数 | 作用 |
|------|------|------|
| 数据增强 | RandomAffine(±10°, ±10%, 0.9~1.1) | 增强泛化能力 |
| 优化器 | AdamW (lr=0.001, weight_decay=1e-4) | L2正则化防过拟合 |
| LR调度 | CosineAnnealingLR | 学习率余弦退火 |
| Dropout | rate=0.3 | 防止过拟合 |
| 早停策略 | 自动保存最佳模型 | 确保最优性能 |

### 实验结果

| Epoch | 测试准确率 |
|-------|-----------|
| Epoch 1 | 97.93% |
| Epoch 2 | 98.37% ✅ 超98% |
| Epoch 3 | 98.83% |
| Epoch 4 | 98.97% |
| Epoch 5 | 99.07% |
| Epoch 6+ | **99.39%+** 🏆 |

---

## 🌐 Web 演示系统

### 技术栈

| 组件 | 技术 | 说明 |
|------|------|------|
| 后端框架 | Flask (Python) | 轻量级Web框架 |
| 前端UI | HTML5 Canvas | 手写数字输入画板 |
| 通信协议 | REST API (JSON) | 前后端分离架构 |
| 推理引擎 | PyTorch | 模型推理 |
| 部署格式 | ONNX | 跨平台模型格式 |

### 架构图

```
浏览器 (Canvas画板)
    │ 用户手写数字
    │ 点击"识别"按钮
    ↓  POST /predict (Base64图片)
Flask Web Server
    │ preprocess_image() 预处理
    │ UltraLightCNN 推理
    ↓  JSON响应 (Top-3预测)
浏览器 (显示结果条)
```

### 使用方法

```bash
# 1. 确保模型已训练
python train_lightweight.py

# 2. 启动服务
python app.py

# 3. 打开浏览器
# 访问: http://127.0.0.1:5000
```

---

## 📁 项目结构

```
├── src/                          # 源代码目录
│   ├── main.py                   MNIST数据加载
│   ├── models.py                 可配置CNN模型
│   ├── train.py                  训练和测试函数
│   ├── advanced_models.py        高级模型（SENet/空洞/深度分离）
│   ├── lightweight_models.py     轻量化模型（UltraLightCNN）
│   ├── adversarial.py            对抗攻击与防御
│   ├── interpretability.py       可解释性分析(Grad-CAM)
│   ├── incremental_learning.py   增量学习(EWC)
│   └── data_augmentation.py      数据增强策略
│
├── experiments/                  # 实验脚本目录
│   ├── run_basic.py              基础实验（层数、核大小对比）
│   ├── run_advanced.py           高级模型对比
│   ├── run_adversarial.py        对抗鲁棒性实验
│   ├── run_interpretability.py   可解释性分析实验
│   └── run_incremental.py        增量学习实验
│
├── templates/
│   └── index.html                Web前端（Canvas画板）
├── app.py                        Flask Web应用
│
├── train_lightweight.py          轻量化模型训练脚本
├── experiments.py                基础CNN对比实验（主脚本）
├── advanced_experiments.py       高级模型对比实验（主脚本）
│
├── experiment_results/           实验输出目录
│   ├── lightweight_cnn.pth       最佳模型权重
│   ├── lightweight_cnn.onnx      ONNX格式模型
│   ├── accuracy_comparison.png   准确率对比图
│   └── analysis_report.txt       分析报告
│
├── results/                      # 分类实验结果
│   ├── basic/                    基础实验结果
│   ├── advanced/                 高级模型结果
│   ├── adversarial/              对抗鲁棒性结果
│   ├── interpretability/         可解释性分析结果
│   └── incremental/              增量学习结果
│
├── docs/                         # 详细文档
│   ├── 基础实验说明.md           基础实验详细说明
│   ├── 非标准卷积与注意力机制说明.md
│   ├── 轻量化与移动端部署说明.md
│   ├── 对抗鲁棒性测试说明.md
│   ├── 可解释性分析说明.md
│   ├── 增量学习与数据增强说明.md
│   └── 项目总体说明.md
│
├── data/                         MNIST数据集（自动下载）
├── requirements.txt              依赖配置文件
└── README.md                     本文档
```

---

## 📊 对比实验体系

### 实验一：卷积层数对性能的影响

| 配置 | 参数量 | 预期准确率 |
|------|--------|-----------|
| 1层卷积 | ~804k | 87-89% |
| 2层卷积 | ~421k | 91-93% |
| 3层卷积 | ~241k | 92-94% |

### 实验二：卷积核大小的影响

| 配置 | 参数量 | 感受野 |
|------|--------|--------|
| 3×3核 | ~421k | 小（细节） |
| 5×5核 | ~455k | 中等 |
| 7×7核 | ~504k | 大（全局） |

### 创新实验：高级模型对比

| 模型 | 技术 | 特点 |
|------|------|------|
| SENet-CNN | 通道注意力 (SE Block) | 增强特征表达 |
| DilatedCNN | 空洞卷积 | 大感受野不增参 |
| DepthwiseCNN | 深度可分离 | 参数量减少75% |

---

## 🔑 关键API

```python
# 数据加载
from src.main import get_data_loaders
train_loader, test_loader = get_data_loaders()

# 基础模型
from src.models import ConfigurableCNN
model = ConfigurableCNN(num_conv_layers=2, kernel_size=3)

# 轻量化模型
from src.lightweight_models import UltraLightCNN
model = UltraLightCNN(dropout_rate=0.3)

# 高级模型
from src.advanced_models import SENetCNN, DilatedCNN, DepthwiseCNN

# 训练函数
from src.train import train_epoch, test, run_training
all_losses, accuracies = run_training(model, train_loader, test_loader, epochs=10)
```

---

## 📝 运行命令速查

```bash
# 快速验证
python quick_test.py

# 基础实验
python experiments.py

# 高级模型实验
python advanced_experiments.py

# 轻量化模型训练
python train_lightweight.py

# Web演示
python app.py

# 检查GPU
python check_gpu.py
```

---

## 📄 License

MIT License
├── requirements.txt         # 项目依赖
├── 使用说明.md             # 详细使用文档
├── 项目总结.md             # 项目完成情况
└── README.md                # 本文档
```

## 📊 输出结果

运行实验后，会在 `experiment_results/` 目录生成：

- � **loss_curve_B.png** - 网络层数的损失曲线
- 📈 **loss_curve_C.png** - 卷积核大小的损失曲线
- 📊 **accuracy_comparison.png** - 准确率对比图
- 📉 **accuracy_curves.png** - 准确率变化曲线
- � **analysis_report.txt** - 详细分析报告
- 💾 **results_*.json** - 实验数据

## 📖 详细文档

本项目提供完整的文档体系，帮助理解和使用各个功能模块：

### 核心文档
- **[基础实验说明](docs/基础实验说明.md)** - 基础CNN实验详细说明（层数、卷积核对比）
- **[项目总体说明](docs/项目总体说明.md)** - 项目整体架构和功能概览

### 创新扩展文档
- **[非标准卷积与注意力机制说明](docs/非标准卷积与注意力机制说明.md)** - SENet、空洞卷积、深度可分离卷积
- **[轻量化与移动端部署说明](docs/轻量化与移动端部署说明.md)** - UltraLightCNN、ONNX导出、Web演示
- **[对抗鲁棒性测试说明](docs/对抗鲁棒性测试说明.md)** - FGSM/PGD攻击与对抗训练
- **[可解释性分析说明](docs/可解释性分析说明.md)** - Grad-CAM热力图可视化
- **[增量学习与数据增强说明](docs/增量学习与数据增强说明.md)** - EWC防遗忘、数据增强策略

## 🎯 预期结果

- **准确率**: 97-99%
- **训练时间（CPU）**: 30-60分钟（完整实验）
- **训练时间（GPU，如RTX 3050）**: 5-10分钟（完整实验）

## 🎮 GPU加速

本项目支持NVIDIA GPU加速训练：

- ✅ 自动检测GPU并使用CUDA加速
- ✅ 支持cuDNN加速
- ✅ 显示GPU信息（型号、显存、CUDA版本）
- ✅ 使用GPU可将训练速度提升5-10倍

**GPU要求**:
- NVIDIA显卡（如RTX 3050、GTX 1060等）
- 安装CUDA版本的PyTorch
- 建议显存 >= 4GB

## � 系统要求

- Python >= 3.7
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- matplotlib
- numpy

## 💡 使用提示

1. **首次运行**: 建议先执行基础实验验证环境配置
2. **训练时间**: 完整实验需要30-60分钟（CPU）或5-10分钟（GPU）
3. **结果输出**: 所有生成的图表和报告可直接用于课设文档
4. **文档查阅**: 每个功能模块都有对应的详细说明文档

## 🤝 贡献与反馈

欢迎提出问题和改进建议！如有疑问，请查阅 `docs/` 目录下的详细文档。

---

**项目状态**: ✅ 完整可用  
**最后更新**: 2026年5月12日
