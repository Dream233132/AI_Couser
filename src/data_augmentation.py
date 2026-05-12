"""
数据增强模块

实现多种创新的数据增强策略：
1. 弹性变形 (Elastic Deformation)
2. 随机擦除 (Random Erasing)
3. MixUp混合增强
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter
from typing import Tuple
import random


class ElasticDeformation:
    """
    弹性变形 (Elastic Deformation)
    
    原理：通过随机位移场对图像进行非线性变形
    特别适合手写数字识别，模拟书写时的自然变化
    
    参数：
        alpha: 变形强度
        sigma: 高斯滤波器标准差，控制变形的平滑度
    """
    
    def __init__(self, alpha: float = 36, sigma: float = 4):
        self.alpha = alpha
        self.sigma = sigma
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        对图像应用弹性变形
        
        参数：
            image: 输入图像 [H, W] 或 [C, H, W]
        
        返回：
            变形后的图像
        """
        if len(image.shape) == 3:
            # [C, H, W] -> [H, W, C]
            image = np.transpose(image, (1, 2, 0))
            is_multi_channel = True
        else:
            is_multi_channel = False
        
        shape = image.shape[:2]
        
        # 生成随机位移场
        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), self.sigma) * self.alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), self.sigma) * self.alpha
        
        # 创建网格坐标
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
        
        # 应用变形
        if is_multi_channel:
            distorted = np.zeros_like(image)
            for i in range(image.shape[2]):
                distorted[:, :, i] = map_coordinates(image[:, :, i], indices, order=1, mode='reflect').reshape(shape)
            # [H, W, C] -> [C, H, W]
            distorted = np.transpose(distorted, (2, 0, 1))
        else:
            distorted = map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
        
        return distorted


class RandomErasing:
    """
    随机擦除 (Random Erasing)
    
    原理：随机选择图像中的矩形区域并擦除（填充随机值或固定值）
    提高模型对遮挡的鲁棒性
    
    参数：
        probability: 应用擦除的概率
        sl: 擦除区域面积下限（相对于图像面积）
        sh: 擦除区域面积上限
        r1: 擦除区域宽高比下限
        mean: 填充值（None表示随机填充）
    """
    
    def __init__(self, probability: float = 0.5, sl: float = 0.02, sh: float = 0.4,
                 r1: float = 0.3, mean: float = None):
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.mean = mean
    
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        对图像应用随机擦除
        
        参数：
            image: 输入图像 [C, H, W]
        
        返回：
            擦除后的图像
        """
        if random.random() > self.probability:
            return image
        
        for attempt in range(100):
            area = image.size(1) * image.size(2)
            
            # 随机选择擦除区域面积
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)
            
            # 计算擦除区域的宽高
            h = int(round(np.sqrt(target_area * aspect_ratio)))
            w = int(round(np.sqrt(target_area / aspect_ratio)))
            
            if w < image.size(2) and h < image.size(1):
                # 随机选择擦除位置
                x1 = random.randint(0, image.size(1) - h)
                y1 = random.randint(0, image.size(2) - w)
                
                # 擦除
                if self.mean is not None:
                    image[:, x1:x1+h, y1:y1+w] = self.mean
                else:
                    image[:, x1:x1+h, y1:y1+w] = torch.rand(image.size(0), h, w)
                
                return image
        
        return image


class MixUp:
    """
    MixUp混合增强
    
    原理：将两个样本及其标签按比例混合
    x_mix = λ * x_i + (1-λ) * x_j
    y_mix = λ * y_i + (1-λ) * y_j
    
    参数：
        alpha: Beta分布参数，控制混合比例的分布
    """
    
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
    
    def __call__(self, images: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对一个batch的图像应用MixUp
        
        参数：
            images: 图像batch [B, C, H, W]
            labels: 标签batch [B]
        
        返回：
            (mixed_images, mixed_labels): 混合后的图像和标签
        """
        batch_size = images.size(0)
        
        # 从Beta分布采样混合比例
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0
        
        # 随机打乱索引
        index = torch.randperm(batch_size).to(images.device)
        
        # 混合图像
        mixed_images = lam * images + (1 - lam) * images[index]
        
        # 转换标签为one-hot编码
        num_classes = 10  # MNIST有10个类别
        labels_a = torch.zeros(batch_size, num_classes).to(images.device)
        labels_b = torch.zeros(batch_size, num_classes).to(images.device)
        labels_a.scatter_(1, labels.unsqueeze(1), 1)
        labels_b.scatter_(1, labels[index].unsqueeze(1), 1)
        
        # 混合标签
        mixed_labels = lam * labels_a + (1 - lam) * labels_b
        
        return mixed_images, mixed_labels


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    MixUp的损失函数
    
    参数：
        criterion: 原始损失函数
        pred: 模型预测
        y_a: 第一个样本的标签
        y_b: 第二个样本的标签
        lam: 混合比例
    
    返回：
        混合损失
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class AugmentedDataset(torch.utils.data.Dataset):
    """
    增强数据集包装器
    
    在数据加载时动态应用数据增强
    """
    
    def __init__(self, dataset, use_elastic: bool = False, use_erasing: bool = False):
        self.dataset = dataset
        self.use_elastic = use_elastic
        self.use_erasing = use_erasing
        
        if use_elastic:
            self.elastic = ElasticDeformation(alpha=36, sigma=4)
        if use_erasing:
            self.erasing = RandomErasing(probability=0.5)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        
        # 应用弹性变形
        if self.use_elastic:
            # 转换为numpy
            img_np = image.numpy()
            img_np = self.elastic(img_np)
            image = torch.from_numpy(img_np).float()
        
        # 应用随机擦除
        if self.use_erasing:
            image = self.erasing(image)
        
        return image, label


def train_with_augmentation(model: nn.Module, train_loader, test_loader,
                           epochs: int, learning_rate: float, device: torch.device,
                           use_mixup: bool = False, mixup_alpha: float = 1.0) -> Tuple:
    """
    使用数据增强训练模型
    
    参数：
        model: 神经网络模型
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        epochs: 训练轮数
        learning_rate: 学习率
        device: 计算设备
        use_mixup: 是否使用MixUp
        mixup_alpha: MixUp的alpha参数
    
    返回：
        (losses, accuracies): 训练损失和测试准确率列表
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    if use_mixup:
        mixup = MixUp(alpha=mixup_alpha)
    
    losses = []
    accuracies = []
    
    print(f"\n{'='*70}")
    print(f"开始数据增强训练 | MixUp={use_mixup} | Epochs={epochs}")
    print(f"{'='*70}\n")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            if use_mixup:
                # 应用MixUp
                mixed_images, mixed_labels = mixup(images, labels)
                outputs = model(mixed_images)
                
                # MixUp损失（使用soft labels）
                loss = -torch.mean(torch.sum(mixed_labels * torch.log_softmax(outputs, dim=1), dim=1))
            else:
                # 标准训练
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if (batch_idx + 1) % 100 == 0:
                print(f"  Epoch [{epoch+1}/{epochs}] Batch [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}")
        
        # 测试
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100.0 * correct / total
        avg_loss = epoch_loss / len(train_loader)
        
        losses.append(avg_loss)
        accuracies.append(accuracy)
        
        print(f"\nEpoch [{epoch+1}/{epochs}]")
        print(f"  Avg Loss: {avg_loss:.4f}")
        print(f"  Test Accuracy: {accuracy:.2f}%\n")
    
    print(f"{'='*70}")
    print("数据增强训练完成！")
    print(f"{'='*70}\n")
    
    return losses, accuracies
