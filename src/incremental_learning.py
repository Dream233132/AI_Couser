"""
增量学习模块

实现EWC (Elastic Weight Consolidation) 方法
用于在学习新任务时防止遗忘旧任务知识
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from typing import Dict, List, Tuple
import numpy as np
import copy


class EWC:
    """
    EWC (Elastic Weight Consolidation) - 弹性权重巩固
    
    原理：
    1. 在旧任务上计算Fisher信息矩阵，衡量参数重要性
    2. 在新任务训练时，添加正则化项约束重要参数不大幅变化
    3. 损失函数：L = L_new + λ/2 * Σ F_i * (θ_i - θ*_i)^2
    
    参数：
        model: 神经网络模型
        dataloader: 旧任务数据加载器
        device: 计算设备
    """
    
    def __init__(self, model: nn.Module, dataloader: DataLoader, device: torch.device):
        self.model = model
        self.device = device
        self.params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
        self.fisher = self._compute_fisher(dataloader)
    
    def _compute_fisher(self, dataloader: DataLoader) -> Dict[str, torch.Tensor]:
        """
        计算Fisher信息矩阵（对角近似）
        
        Fisher信息矩阵衡量参数对损失函数的敏感度
        F_i = E[(∂log p(y|x,θ) / ∂θ_i)^2]
        
        参数：
            dataloader: 数据加载器
        
        返回：
            fisher: Fisher信息矩阵字典 {param_name: fisher_value}
        """
        self.model.eval()
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}
        
        print("\n计算Fisher信息矩阵...")
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # 前向传播
            self.model.zero_grad()
            outputs = self.model(images)
            
            # 计算负对数似然损失
            loss = F.cross_entropy(outputs, labels)
            
            # 反向传播获取梯度
            loss.backward()
            
            # 累积梯度平方（Fisher信息矩阵的对角近似）
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.pow(2)
            
            if (batch_idx + 1) % 100 == 0:
                print(f"  处理批次 [{batch_idx + 1}/{len(dataloader)}]")
        
        # 归一化（除以样本数）
        num_samples = len(dataloader.dataset)
        for n in fisher:
            fisher[n] /= num_samples
        
        print("✓ Fisher信息矩阵计算完成\n")
        
        return fisher
    
    def penalty(self, model: nn.Module) -> torch.Tensor:
        """
        计算EWC正则化惩罚项
        
        penalty = λ/2 * Σ F_i * (θ_i - θ*_i)^2
        
        参数：
            model: 当前模型
        
        返回：
            loss: EWC惩罚损失
        """
        loss = 0.0
        for n, p in model.named_parameters():
            if n in self.fisher and p.requires_grad:
                # Fisher加权的参数变化平方
                loss += (self.fisher[n] * (p - self.params[n]).pow(2)).sum()
        
        return loss


def train_with_ewc(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader,
                   ewc: EWC, epochs: int, learning_rate: float, ewc_lambda: float,
                   device: torch.device) -> Tuple[List[float], List[float]]:
    """
    使用EWC进行增量学习训练
    
    参数：
        model: 神经网络模型
        train_loader: 新任务训练数据
        test_loader: 测试数据
        ewc: EWC对象（包含旧任务的Fisher信息）
        epochs: 训练轮数
        learning_rate: 学习率
        ewc_lambda: EWC正则化系数
        device: 计算设备
    
    返回：
        (losses, accuracies): 训练损失和测试准确率列表
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    losses = []
    accuracies = []
    
    print(f"\n{'='*70}")
    print(f"开始EWC增量学习训练 | λ={ewc_lambda} | Epochs={epochs}")
    print(f"{'='*70}\n")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(images)
            
            # 计算任务损失
            task_loss = criterion(outputs, labels)
            
            # 计算EWC惩罚
            ewc_loss = ewc.penalty(model) if ewc is not None else 0.0
            
            # 总损失
            loss = task_loss + (ewc_lambda / 2) * ewc_loss
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if (batch_idx + 1) % 100 == 0:
                print(f"  Epoch [{epoch+1}/{epochs}] Batch [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f} (Task: {task_loss.item():.4f}, EWC: {ewc_loss if isinstance(ewc_loss, float) else ewc_loss.item():.4f})")
        
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
    print("EWC训练完成！")
    print(f"{'='*70}\n")
    
    return losses, accuracies


def evaluate_forgetting(model: nn.Module, old_task_loader: DataLoader, 
                       new_task_loader: DataLoader, device: torch.device) -> Dict:
    """
    评估模型在旧任务和新任务上的性能，计算遗忘率
    
    参数：
        model: 训练后的模型
        old_task_loader: 旧任务测试数据
        new_task_loader: 新任务测试数据
        device: 计算设备
    
    返回：
        results: 包含准确率和遗忘率的字典
    """
    model.eval()
    
    def evaluate_on_loader(loader):
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100.0 * correct / total
    
    old_acc = evaluate_on_loader(old_task_loader)
    new_acc = evaluate_on_loader(new_task_loader)
    
    return {
        'old_task_accuracy': old_acc,
        'new_task_accuracy': new_acc
    }


def incremental_learning_experiment(model_class, train_dataset, test_dataset,
                                   old_classes: List[int], new_classes: List[int],
                                   epochs: int, learning_rate: float, ewc_lambda: float,
                                   device: torch.device, use_ewc: bool = True) -> Dict:
    """
    完整的增量学习实验
    
    场景：先训练旧类别，再学习新类别
    
    参数：
        model_class: 模型类
        train_dataset: 训练数据集
        test_dataset: 测试数据集
        old_classes: 旧任务类别列表（如[0,1,2,3,4]）
        new_classes: 新任务类别列表（如[5,6,7,8,9]）
        epochs: 每个任务的训练轮数
        learning_rate: 学习率
        ewc_lambda: EWC正则化系数
        device: 计算设备
        use_ewc: 是否使用EWC
    
    返回：
        results: 实验结果字典
    """
    print("\n" + "="*70)
    print("增量学习实验")
    print("="*70)
    print(f"旧任务类别: {old_classes}")
    print(f"新任务类别: {new_classes}")
    print(f"使用EWC: {use_ewc}")
    print("="*70 + "\n")
    
    # 1. 准备旧任务数据
    old_train_indices = [i for i, (_, label) in enumerate(train_dataset) if label in old_classes]
    old_test_indices = [i for i, (_, label) in enumerate(test_dataset) if label in old_classes]
    
    old_train_loader = DataLoader(Subset(train_dataset, old_train_indices), 
                                  batch_size=64, shuffle=True)
    old_test_loader = DataLoader(Subset(test_dataset, old_test_indices), 
                                 batch_size=1000, shuffle=False)
    
    # 2. 训练旧任务
    print("阶段1: 训练旧任务...")
    model = model_class().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        for batch_idx, (images, labels) in enumerate(old_train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            if (batch_idx + 1) % 100 == 0:
                print(f"  Epoch [{epoch+1}/{epochs}] Batch [{batch_idx+1}/{len(old_train_loader)}] Loss: {loss.item():.4f}")
        
        # 测试
        old_acc = evaluate_forgetting(model, old_test_loader, old_test_loader, device)['old_task_accuracy']
        print(f"  Epoch [{epoch+1}/{epochs}] 旧任务准确率: {old_acc:.2f}%\n")
    
    old_task_acc_before = evaluate_forgetting(model, old_test_loader, old_test_loader, device)['old_task_accuracy']
    print(f"✓ 旧任务训练完成，准确率: {old_task_acc_before:.2f}%\n")
    
    # 3. 计算Fisher信息矩阵（如果使用EWC）
    ewc = None
    if use_ewc:
        ewc = EWC(model, old_train_loader, device)
    
    # 4. 准备新任务数据
    new_train_indices = [i for i, (_, label) in enumerate(train_dataset) if label in new_classes]
    new_test_indices = [i for i, (_, label) in enumerate(test_dataset) if label in new_classes]
    
    new_train_loader = DataLoader(Subset(train_dataset, new_train_indices), 
                                  batch_size=64, shuffle=True)
    new_test_loader = DataLoader(Subset(test_dataset, new_test_indices), 
                                 batch_size=1000, shuffle=False)
    
    # 5. 训练新任务
    print("阶段2: 训练新任务...")
    if use_ewc:
        train_with_ewc(model, new_train_loader, new_test_loader, ewc, 
                      epochs, learning_rate, ewc_lambda, device)
    else:
        # 不使用EWC，直接训练
        for epoch in range(epochs):
            model.train()
            for batch_idx, (images, labels) in enumerate(new_train_loader):
                images = images.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                if (batch_idx + 1) % 100 == 0:
                    print(f"  Epoch [{epoch+1}/{epochs}] Batch [{batch_idx+1}/{len(new_train_loader)}] Loss: {loss.item():.4f}")
            
            new_acc = evaluate_forgetting(model, new_test_loader, new_test_loader, device)['old_task_accuracy']
            print(f"  Epoch [{epoch+1}/{epochs}] 新任务准确率: {new_acc:.2f}%\n")
    
    # 6. 最终评估
    print("最终评估...")
    results = evaluate_forgetting(model, old_test_loader, new_test_loader, device)
    
    old_task_acc_after = results['old_task_accuracy']
    new_task_acc = results['new_task_accuracy']
    forgetting = old_task_acc_before - old_task_acc_after
    
    print(f"\n{'='*70}")
    print("实验结果")
    print(f"{'='*70}")
    print(f"旧任务准确率（训练前）: {old_task_acc_before:.2f}%")
    print(f"旧任务准确率（训练后）: {old_task_acc_after:.2f}%")
    print(f"新任务准确率: {new_task_acc:.2f}%")
    print(f"遗忘率: {forgetting:.2f}%")
    print(f"{'='*70}\n")
    
    return {
        'old_task_acc_before': old_task_acc_before,
        'old_task_acc_after': old_task_acc_after,
        'new_task_acc': new_task_acc,
        'forgetting': forgetting,
        'use_ewc': use_ewc
    }
