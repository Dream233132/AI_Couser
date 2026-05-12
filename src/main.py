"""
PyTorch MNIST 数据集加载和预处理

本模块提供加载和预处理MNIST数据集的功能，包括：
- 使用torchvision.datasets.MNIST下载数据集
- 数据标准化处理
- 创建优化的DataLoader
"""

import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_data_loaders(batch_size_train=64, batch_size_test=1000, data_dir='./data'):
    """
    加载和预处理MNIST数据集，返回训练和测试的DataLoader
    
    参数说明：
        batch_size_train (int): 训练集batch大小，默认64
        batch_size_test (int): 测试集batch大小，默认1000
        data_dir (str): 数据存储目录，默认为'./data'
    
    返回值：
        tuple: (train_loader, test_loader)
            - train_loader: 训练集DataLoader
            - test_loader: 测试集DataLoader
    
    示例：
        >>> train_loader, test_loader = get_data_loaders()
        >>> for images, labels in train_loader:
        ...     print(images.shape)  # 输出: torch.Size([64, 1, 28, 28])
        ...     break
    """
    
    # 创建数据目录
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # ============================================
    # 第一步：定义数据预处理管道（Transforms）
    # ============================================
    # MNIST数据集的标准统计参数
    MNIST_MEAN = 0.1307  # MNIST训练集的平均值
    MNIST_STD = 0.3081   # MNIST训练集的标准差
    
    # 训练集预处理：包括数据增强和标准化
    train_transform = transforms.Compose([
        # 1. 将PIL图像或numpy数组转换为Tensor
        # 图像值范围从[0, 255]自动转换到[0.0, 1.0]
        transforms.ToTensor(),
        
        # 2. 标准化处理
        # 使用MNIST数据集的标准均值和标准差进行标准化
        # 标准化公式：(x - mean) / std
        transforms.Normalize(mean=MNIST_MEAN, std=MNIST_STD),
    ])
    
    # 测试集预处理：与训练集相同，保持一致性
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=MNIST_MEAN, std=MNIST_STD),
    ])
    
    # ============================================
    # 第二步：下载并加载MNIST数据集
    # ============================================
    print("正在加载MNIST训练集...")
    train_dataset = datasets.MNIST(
        root=data_dir,           # 数据存储目录
        train=True,              # 加载训练集
        download=True,           # 如果数据不存在则下载
        transform=train_transform # 应用预处理管道
    )
    
    print("正在加载MNIST测试集...")
    test_dataset = datasets.MNIST(
        root=data_dir,           # 数据存储目录
        train=False,             # 加载测试集
        download=True,           # 如果数据不存在则下载
        transform=test_transform # 应用预处理管道
    )
    
    # ============================================
    # 第三步：创建DataLoader
    # ============================================
    # 训练集DataLoader：启用shuffle打乱数据，num_workers提高加载效率
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,  # 每批处理64个样本
        shuffle=True,                 # 打乱数据顺序，有利于模型训练
        num_workers=0,                # Windows下建议设为0，避免多进程问题
        pin_memory=True if torch.cuda.is_available() else False  # GPU下提高传输速度
    )
    
    # 测试集DataLoader：不shuffle，保持原始顺序便于结果分析
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size_test,   # 每批处理1000个样本（整个测试集）
        shuffle=False,                # 不打乱顺序
        num_workers=0,                # Windows下建议设为0
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, test_loader


def inspect_data(train_loader):
    """
    查看数据集的基本信息
    
    参数说明：
        train_loader: 训练集DataLoader
    
    打印内容：
        - 单个批次的图像张量形状
        - 图像的数据类型
        - 标签的数据类型
        - 图像的值域范围
    """
    print("\n" + "="*50)
    print("MNIST数据集信息")
    print("="*50)
    
    # 获取第一个batch的数据
    images, labels = next(iter(train_loader))
    
    print(f"\n单个batch的图像张量形状: {images.shape}")
    print(f"  - batch_size (批大小): {images.shape[0]}")
    print(f"  - channels (通道数): {images.shape[1]}")
    print(f"  - height (高): {images.shape[2]}")
    print(f"  - width (宽): {images.shape[3]}")
    
    print(f"\n标签张量形状: {labels.shape}")
    print(f"标签类型: {labels.dtype}")
    
    print(f"\n图像数据类型: {images.dtype}")
    print(f"图像值域范围: [{images.min():.4f}, {images.max():.4f}]")
    print(f"图像的平均值: {images.mean():.4f}")
    print(f"图像的标准差: {images.std():.4f}")
    
    print(f"\n标签示例: {labels[:10].tolist()}")
    print("="*50 + "\n")


def main():
    """
    主程序：演示如何使用DataLoader
    """
    print("PyTorch MNIST 数据加载示例")
    print("="*50)
    
    # 加载数据
    train_loader, test_loader = get_data_loaders()
    
    # 查看数据信息
    inspect_data(train_loader)
    
    # 输出统计信息
    print(f"训练集batch数量: {len(train_loader)}")
    print(f"测试集batch数量: {len(test_loader)}")
    print(f"训练集总样本数: {len(train_loader.dataset)}")
    print(f"测试集总样本数: {len(test_loader.dataset)}")
    
    print("\n✓ 数据加载完成！可以开始模型训练。")
    print("  使用方法: train_loader, test_loader = get_data_loaders()")


if __name__ == "__main__":
    main()


