"""
MNIST 图像分类训练和测试模块

提供完整的训练、测试和评估函数，支持：
- 单个 epoch 的训练（返回所有 batch 的 loss）
- 测试集评估（返回平均 loss 和准确率）
- 完整的训练循环（返回所有迭代的 loss 和每个 epoch 的测试准确率）
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from typing import Tuple, List


def train_epoch(
    model: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module
) -> List[float]:
    """
    训练一个 epoch，返回所有 batch 的 loss 值列表
    
    参数说明：
        model (nn.Module): 神经网络模型
        device (torch.device): 计算设备（CPU 或 GPU）
        train_loader (DataLoader): 训练集数据加载器
        optimizer (torch.optim.Optimizer): 优化器（如 Adam）
        criterion (nn.Module): 损失函数（如 CrossEntropyLoss）
    
    返回值：
        list: 包含每个 batch 的损失值，用于绘制训练曲线
        - 元素为 float 类型，代表该 batch 的损失
        - 长度等于训练集中的 batch 总数
    
    示例：
        >>> model = ConfigurableCNN()
        >>> optimizer = Adam(model.parameters(), lr=0.001)
        >>> criterion = nn.CrossEntropyLoss()
        >>> model = model.to(device)
        >>> iteration_losses = train_epoch(model, device, train_loader, optimizer, criterion)
        >>> print(f"一个epoch中有 {len(iteration_losses)} 个batch")
        >>> print(f"最后一个batch的loss: {iteration_losses[-1]:.4f}")
    """
    
    # 设置模型为训练模式
    model.train()
    
    # 用于保存每个 iteration（batch）的 loss
    iteration_losses = []
    
    # 迭代训练集中的每个 batch
    for batch_idx, (images, labels) in enumerate(train_loader):
        # 1. 将数据移动到指定设备（CPU 或 GPU）
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # 2. 清除梯度
        optimizer.zero_grad()
        
        # 3. 前向传播
        outputs = model(images)
        
        # 4. 计算损失
        loss = criterion(outputs, labels)
        
        # 5. 反向传播
        loss.backward()
        
        # 6. 更新参数
        optimizer.step()
        
        # 7. 记录该 batch 的 loss 值
        iteration_losses.append(loss.item())
        
        # 8. 清理GPU缓存（如果使用GPU）
        if device.type == 'cuda':
            # 每100个batch清理一次，避免内存碎片
            if (batch_idx + 1) % 100 == 0:
                torch.cuda.empty_cache()
        
        # 打印进度信息（可选）
        if (batch_idx + 1) % 100 == 0:
            print(f"  Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")
    
    return iteration_losses


def test(
    model: nn.Module,
    device: torch.device,
    test_loader: DataLoader,
    criterion: nn.Module
) -> Tuple[float, float]:
    """
    在测试集上评估模型性能
    
    参数说明：
        model (nn.Module): 神经网络模型
        device (torch.device): 计算设备（CPU 或 GPU）
        test_loader (DataLoader): 测试集数据加载器
        criterion (nn.Module): 损失函数（如 CrossEntropyLoss）
    
    返回值：
        tuple: (avg_loss, accuracy)
            - avg_loss (float): 平均损失值
            - accuracy (float): 准确率（百分比形式，0-100）
    
    示例：
        >>> model = ConfigurableCNN()
        >>> criterion = nn.CrossEntropyLoss()
        >>> model = model.to(device)
        >>> avg_loss, accuracy = test(model, device, test_loader, criterion)
        >>> print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    """
    
    # 设置模型为评估模式
    model.eval()
    
    # 用于累计 loss 和正确预测数
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    # 在评估过程中不计算梯度，以节省内存和加速
    with torch.no_grad():
        for images, labels in test_loader:
            # 1. 将数据移动到指定设备
            images = images.to(device)
            labels = labels.to(device)
            
            # 2. 前向传播
            outputs = model(images)
            
            # 3. 计算损失
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            
            # 4. 计算预测准确率
            # outputs 的形状是 (batch_size, num_classes)
            # argmax 返回每行的最大值索引（预测类别）
            _, predicted = torch.max(outputs, 1)
            
            # 统计预测正确的样本数
            total_correct += (predicted == labels).sum().item()
            
            # 统计样本总数
            total_samples += labels.size(0)
    
    # 计算平均 loss 和准确率
    avg_loss = total_loss / total_samples
    accuracy = (total_correct / total_samples) * 100  # 转换为百分比
    
    return avg_loss, accuracy


def run_training(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 10,
    learning_rate: float = 0.001,
    device: torch.device = None
) -> Tuple[List[float], List[float]]:
    """
    完整的训练循环控制函数
    
    使用 Adam 优化器和 CrossEntropyLoss，在给定的 epoch 内循环调用 train 和 test。
    返回训练过程中所有 batch 的损失和每个 epoch 的测试准确率。
    
    参数说明：
        model (nn.Module): 神经网络模型
        train_loader (DataLoader): 训练集数据加载器
        test_loader (DataLoader): 测试集数据加载器
        epochs (int): 训练的 epoch 数量，默认 10
        learning_rate (float): Adam 优化器的学习率，默认 0.001
        device (torch.device): 计算设备（CPU 或 GPU），默认自动选择
    
    返回值：
        tuple: (all_iteration_losses, epoch_test_accuracies)
            - all_iteration_losses (list): 包含训练过程中所有 batch 的损失
              * 长度 = sum(每个epoch的batch数)
              * 用于绘制平滑的训练损失曲线
            - epoch_test_accuracies (list): 每个 epoch 结束后的测试准确率
              * 长度 = epochs
              * 单位为百分比（0-100）
    
    示例：
        >>> from src.main import get_data_loaders
        >>> from src.models import ConfigurableCNN
        >>> import torch
        >>> 
        >>> device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        >>> train_loader, test_loader = get_data_loaders(batch_size_train=64, batch_size_test=1000)
        >>> model = ConfigurableCNN(num_conv_layers=2, kernel_size=3)
        >>> 
        >>> all_losses, accuracies = run_training(
        ...     model=model,
        ...     train_loader=train_loader,
        ...     test_loader=test_loader,
        ...     epochs=5,
        ...     learning_rate=0.001,
        ...     device=device
        ... )
        >>> 
        >>> print(f"总共记录了 {len(all_losses)} 个 batch 的 loss")
        >>> print(f"测试准确率: {accuracies}")
    """
    
    # 1. 如果未指定设备，自动选择
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 2. 将模型移动到指定设备
    model = model.to(device)
    
    # 3. 初始化优化器（Adam）
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    # 4. 初始化损失函数（CrossEntropyLoss）
    criterion = nn.CrossEntropyLoss()
    
    # 5. 初始化返回值列表
    all_iteration_losses = []  # 保存所有 batch 的 loss
    epoch_test_accuracies = []  # 保存每个 epoch 的测试准确率
    
    # 6. 训练循环
    print(f"\n{'='*70}")
    print(f"开始训练 | 设备: {device} | 学习率: {learning_rate} | Epochs: {epochs}")
    print(f"{'='*70}\n")
    
    for epoch in range(epochs):
        print(f"Epoch [{epoch + 1}/{epochs}]")
        print("-" * 70)
        
        # 训练一个 epoch
        iteration_losses = train_epoch(model, device, train_loader, optimizer, criterion)
        
        # 将该 epoch 的所有 loss 添加到总列表中
        all_iteration_losses.extend(iteration_losses)
        
        # 在测试集上评估
        test_loss, test_accuracy = test(model, device, test_loader, criterion)
        
        # 记录该 epoch 的测试准确率
        epoch_test_accuracies.append(test_accuracy)
        
        # 打印该 epoch 的统计信息
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Test Accuracy: {test_accuracy:.2f}%")
        print()
    
    print(f"{'='*70}")
    print(f"训练完成!")
    print(f"总 Batch 数: {len(all_iteration_losses)}")
    print(f"最终测试准确率: {epoch_test_accuracies[-1]:.2f}%")
    print(f"{'='*70}\n")
    
    return all_iteration_losses, epoch_test_accuracies


# ============================================================================
# 辅助函数：用于分析和可视化训练结果
# ============================================================================

def smooth_curve(values: List[float], window_size: int = 20) -> List[float]:
    """
    对曲线进行平滑处理（移动平均）
    
    参数说明：
        values (list): 原始数据点列表
        window_size (int): 移动平均窗口大小，默认 20
    
    返回值：
        list: 平滑后的数据点
    
    示例：
        >>> losses = [0.5, 0.48, 0.52, 0.49, 0.47, 0.46]
        >>> smoothed = smooth_curve(losses, window_size=2)
    """
    import numpy as np
    
    if len(values) < window_size:
        return values
    
    smoothed = []
    for i in range(len(values)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(values), i + window_size // 2 + 1)
        smoothed.append(np.mean(values[start_idx:end_idx]))
    
    return smoothed


def print_training_summary(
    all_iteration_losses: List[float],
    epoch_test_accuracies: List[float]
) -> None:
    """
    打印训练总结统计信息
    
    参数说明：
        all_iteration_losses (list): 所有 batch 的 loss
        epoch_test_accuracies (list): 每个 epoch 的测试准确率
    
    示例：
        >>> print_training_summary(all_losses, accuracies)
    """
    import numpy as np
    
    print("\n" + "="*70)
    print("训练总结")
    print("="*70)
    
    print("\n【训练 Loss】")
    print(f"  初始 Loss:     {all_iteration_losses[0]:.4f}")
    print(f"  最终 Loss:     {all_iteration_losses[-1]:.4f}")
    print(f"  平均 Loss:     {np.mean(all_iteration_losses):.4f}")
    print(f"  最小 Loss:     {np.min(all_iteration_losses):.4f}")
    print(f"  最大 Loss:     {np.max(all_iteration_losses):.4f}")
    
    print("\n【测试准确率】")
    print(f"  初始准确率:    {epoch_test_accuracies[0]:.2f}%")
    print(f"  最终准确率:    {epoch_test_accuracies[-1]:.2f}%")
    print(f"  平均准确率:    {np.mean(epoch_test_accuracies):.2f}%")
    print(f"  最高准确率:    {np.max(epoch_test_accuracies):.2f}%")
    
    # 计算准确率的改进
    accuracy_improvement = epoch_test_accuracies[-1] - epoch_test_accuracies[0]
    print(f"  准确率提升:    {accuracy_improvement:+.2f}%")
    
    print("\n" + "="*70 + "\n")
