"""
对抗鲁棒性测试模块

实现对抗攻击方法和防御策略：
1. FGSM (Fast Gradient Sign Method) - 单步对抗攻击
2. PGD (Projected Gradient Descent) - 多步迭代对抗攻击
3. 对抗训练 (Adversarial Training) - 防御方法
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
import numpy as np


class FGSMAttack:
    """
    FGSM (Fast Gradient Sign Method) 对抗攻击
    
    原理：在输入图像上添加扰动，扰动方向为损失函数梯度的符号方向
    公式：x_adv = x + ε * sign(∇_x L(θ, x, y))
    
    参数：
        epsilon (float): 扰动强度，控制对抗样本与原始样本的差异
    """
    
    def __init__(self, epsilon: float = 0.1):
        self.epsilon = epsilon
    
    def generate(self, model: nn.Module, images: torch.Tensor, 
                 labels: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        生成FGSM对抗样本
        
        参数：
            model: 目标模型
            images: 原始图像 [batch_size, channels, height, width]
            labels: 真实标签 [batch_size]
            device: 计算设备
        
        返回：
            对抗样本 [batch_size, channels, height, width]
        """
        # 将数据移到设备
        images = images.to(device)
        labels = labels.to(device)
        
        # 需要计算梯度
        images.requires_grad = True
        
        # 前向传播
        outputs = model(images)
        
        # 计算损失
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, labels)
        
        # 反向传播获取梯度
        model.zero_grad()
        loss.backward()
        
        # 获取数据梯度
        data_grad = images.grad.data
        
        # 生成对抗样本：x_adv = x + ε * sign(∇_x L)
        perturbed_images = images + self.epsilon * data_grad.sign()
        
        # 裁剪到有效范围 [0, 1]（假设输入已归一化）
        perturbed_images = torch.clamp(perturbed_images, 0, 1)
        
        return perturbed_images.detach()


class PGDAttack:
    """
    PGD (Projected Gradient Descent) 对抗攻击
    
    原理：多步迭代的FGSM，每步添加小扰动并投影回ε-球内
    是目前最强的一阶对抗攻击方法
    
    参数：
        epsilon (float): 最大扰动强度
        alpha (float): 每步的步长
        num_iter (int): 迭代次数
    """
    
    def __init__(self, epsilon: float = 0.1, alpha: float = 0.01, num_iter: int = 40):
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_iter = num_iter
    
    def generate(self, model: nn.Module, images: torch.Tensor, 
                 labels: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        生成PGD对抗样本
        
        参数：
            model: 目标模型
            images: 原始图像
            labels: 真实标签
            device: 计算设备
        
        返回：
            对抗样本
        """
        images = images.to(device)
        labels = labels.to(device)
        
        # 从原始图像开始，添加随机噪声初始化
        perturbed_images = images.clone().detach()
        perturbed_images = perturbed_images + torch.empty_like(perturbed_images).uniform_(-self.epsilon, self.epsilon)
        perturbed_images = torch.clamp(perturbed_images, 0, 1)
        
        # 迭代攻击
        for i in range(self.num_iter):
            perturbed_images.requires_grad = True
            
            # 前向传播
            outputs = model(perturbed_images)
            
            # 计算损失
            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs, labels)
            
            # 反向传播
            model.zero_grad()
            loss.backward()
            
            # 获取梯度
            data_grad = perturbed_images.grad.data
            
            # 更新对抗样本：x_adv = x_adv + α * sign(∇_x L)
            perturbed_images = perturbed_images.detach() + self.alpha * data_grad.sign()
            
            # 投影回ε-球内
            delta = torch.clamp(perturbed_images - images, min=-self.epsilon, max=self.epsilon)
            perturbed_images = torch.clamp(images + delta, 0, 1).detach()
        
        return perturbed_images


def evaluate_robustness(model: nn.Module, test_loader, attack, 
                       device: torch.device) -> Tuple[float, float]:
    """
    评估模型在对抗攻击下的鲁棒性
    
    参数：
        model: 待测试模型
        test_loader: 测试数据加载器
        attack: 攻击方法（FGSM或PGD实例）
        device: 计算设备
    
    返回：
        (clean_accuracy, adversarial_accuracy): 干净样本准确率和对抗样本准确率
    """
    model.eval()
    
    clean_correct = 0
    adv_correct = 0
    total = 0
    
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        # 1. 在干净样本上测试
        with torch.no_grad():
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            clean_correct += (predicted == labels).sum().item()
        
        # 2. 生成对抗样本
        adv_images = attack.generate(model, images, labels, device)
        
        # 3. 在对抗样本上测试
        with torch.no_grad():
            outputs = model(adv_images)
            _, predicted = torch.max(outputs, 1)
            adv_correct += (predicted == labels).sum().item()
        
        total += labels.size(0)
    
    clean_accuracy = 100.0 * clean_correct / total
    adv_accuracy = 100.0 * adv_correct / total
    
    return clean_accuracy, adv_accuracy


def adversarial_training(model: nn.Module, train_loader, test_loader,
                        epochs: int, learning_rate: float, epsilon: float,
                        device: torch.device) -> Tuple[List[float], List[float]]:
    """
    对抗训练 - 提升模型鲁棒性的防御方法
    
    原理：在训练过程中混入对抗样本，让模型学习对抗扰动的鲁棒特征
    
    参数：
        model: 待训练模型
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        epochs: 训练轮数
        learning_rate: 学习率
        epsilon: 对抗扰动强度
        device: 计算设备
    
    返回：
        (clean_accuracies, adv_accuracies): 每个epoch的干净/对抗准确率列表
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # 使用FGSM生成对抗样本
    fgsm = FGSMAttack(epsilon=epsilon)
    
    clean_accuracies = []
    adv_accuracies = []
    
    print(f"\n{'='*70}")
    print(f"开始对抗训练 | ε={epsilon} | Epochs={epochs}")
    print(f"{'='*70}\n")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            # 生成对抗样本
            adv_images = fgsm.generate(model, images, labels, device)
            
            # 混合训练：50%干净样本 + 50%对抗样本
            mixed_images = torch.cat([images, adv_images], dim=0)
            mixed_labels = torch.cat([labels, labels], dim=0)
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(mixed_images)
            loss = criterion(outputs, mixed_labels)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if (batch_idx + 1) % 100 == 0:
                print(f"  Epoch [{epoch+1}/{epochs}] Batch [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}")
        
        # 评估
        clean_acc, adv_acc = evaluate_robustness(model, test_loader, fgsm, device)
        clean_accuracies.append(clean_acc)
        adv_accuracies.append(adv_acc)
        
        print(f"\nEpoch [{epoch+1}/{epochs}]")
        print(f"  Clean Accuracy: {clean_acc:.2f}%")
        print(f"  Adversarial Accuracy: {adv_acc:.2f}%")
        print(f"  Avg Loss: {train_loss/len(train_loader):.4f}\n")
    
    print(f"{'='*70}")
    print("对抗训练完成！")
    print(f"{'='*70}\n")
    
    return clean_accuracies, adv_accuracies


def generate_adversarial_examples(model: nn.Module, test_loader, 
                                  epsilon: float, device: torch.device,
                                  num_examples: int = 10) -> Tuple:
    """
    生成对抗样本用于可视化
    
    参数：
        model: 目标模型
        test_loader: 测试数据加载器
        epsilon: 扰动强度
        device: 计算设备
        num_examples: 生成样本数量
    
    返回：
        (original_images, adv_images, original_labels, original_preds, adv_preds)
    """
    model.eval()
    fgsm = FGSMAttack(epsilon=epsilon)
    
    original_images = []
    adv_images = []
    original_labels = []
    original_preds = []
    adv_preds = []
    
    for images, labels in test_loader:
        if len(original_images) >= num_examples:
            break
        
        images = images.to(device)
        labels = labels.to(device)
        
        # 原始预测
        with torch.no_grad():
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
        
        # 生成对抗样本
        adv_imgs = fgsm.generate(model, images, labels, device)
        
        # 对抗样本预测
        with torch.no_grad():
            adv_outputs = model(adv_imgs)
            _, adv_preds_batch = torch.max(adv_outputs, 1)
        
        # 收集结果
        for i in range(min(images.size(0), num_examples - len(original_images))):
            original_images.append(images[i].cpu())
            adv_images.append(adv_imgs[i].cpu())
            original_labels.append(labels[i].cpu().item())
            original_preds.append(preds[i].cpu().item())
            adv_preds.append(adv_preds_batch[i].cpu().item())
    
    return (torch.stack(original_images), torch.stack(adv_images), 
            original_labels, original_preds, adv_preds)
