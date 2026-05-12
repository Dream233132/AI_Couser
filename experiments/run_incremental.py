"""
增量学习与数据增强实验

测试EWC增量学习方法和多种数据增强策略的效果
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

from src.main import get_data_loaders
from src.models import ConfigurableCNN
from src.incremental_learning import incremental_learning_experiment
from src.data_augmentation import (AugmentedDataset, train_with_augmentation,
                                   ElasticDeformation, RandomErasing)
from torchvision import datasets, transforms

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def test_incremental_learning():
    """测试增量学习：有EWC vs 无EWC"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载完整数据集
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # 定义旧任务和新任务
    old_classes = [0, 1, 2, 3, 4]
    new_classes = [5, 6, 7, 8, 9]
    
    # 实验1: 不使用EWC
    print("\n实验1: 不使用EWC的增量学习...")
    results_no_ewc = incremental_learning_experiment(
        model_class=lambda: ConfigurableCNN(num_conv_layers=2, kernel_size=3),
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        old_classes=old_classes,
        new_classes=new_classes,
        epochs=5,
        learning_rate=0.001,
        ewc_lambda=5000,
        device=device,
        use_ewc=False
    )
    
    # 实验2: 使用EWC
    print("\n实验2: 使用EWC的增量学习...")
    results_with_ewc = incremental_learning_experiment(
        model_class=lambda: ConfigurableCNN(num_conv_layers=2, kernel_size=3),
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        old_classes=old_classes,
        new_classes=new_classes,
        epochs=5,
        learning_rate=0.001,
        ewc_lambda=5000,
        device=device,
        use_ewc=True
    )
    
    return results_no_ewc, results_with_ewc


def test_data_augmentation():
    """测试不同数据增强策略的效果"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载基础数据集
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    results = {}
    
    # 1. 无增强（基线）
    print("\n1. 训练无数据增强的模型（基线）...")
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    model_baseline = ConfigurableCNN(num_conv_layers=2, kernel_size=3)
    _, accs_baseline = train_with_augmentation(
        model_baseline, train_loader, test_loader,
        epochs=5, learning_rate=0.001, device=device, use_mixup=False
    )
    results['无增强'] = accs_baseline
    
    # 2. 弹性变形
    print("\n2. 训练使用弹性变形的模型...")
    aug_dataset_elastic = AugmentedDataset(train_dataset, use_elastic=True, use_erasing=False)
    train_loader_elastic = DataLoader(aug_dataset_elastic, batch_size=64, shuffle=True)
    model_elastic = ConfigurableCNN(num_conv_layers=2, kernel_size=3)
    _, accs_elastic = train_with_augmentation(
        model_elastic, train_loader_elastic, test_loader,
        epochs=5, learning_rate=0.001, device=device, use_mixup=False
    )
    results['弹性变形'] = accs_elastic
    
    # 3. 随机擦除
    print("\n3. 训练使用随机擦除的模型...")
    aug_dataset_erasing = AugmentedDataset(train_dataset, use_elastic=False, use_erasing=True)
    train_loader_erasing = DataLoader(aug_dataset_erasing, batch_size=64, shuffle=True)
    model_erasing = ConfigurableCNN(num_conv_layers=2, kernel_size=3)
    _, accs_erasing = train_with_augmentation(
        model_erasing, train_loader_erasing, test_loader,
        epochs=5, learning_rate=0.001, device=device, use_mixup=False
    )
    results['随机擦除'] = accs_erasing
    
    # 4. MixUp
    print("\n4. 训练使用MixUp的模型...")
    train_loader_mixup = DataLoader(train_dataset, batch_size=64, shuffle=True)
    model_mixup = ConfigurableCNN(num_conv_layers=2, kernel_size=3)
    _, accs_mixup = train_with_augmentation(
        model_mixup, train_loader_mixup, test_loader,
        epochs=5, learning_rate=0.001, device=device, use_mixup=True, mixup_alpha=1.0
    )
    results['MixUp'] = accs_mixup
    
    # 5. 组合增强
    print("\n5. 训练使用组合增强的模型...")
    aug_dataset_combined = AugmentedDataset(train_dataset, use_elastic=True, use_erasing=True)
    train_loader_combined = DataLoader(aug_dataset_combined, batch_size=64, shuffle=True)
    model_combined = ConfigurableCNN(num_conv_layers=2, kernel_size=3)
    _, accs_combined = train_with_augmentation(
        model_combined, train_loader_combined, test_loader,
        epochs=5, learning_rate=0.001, device=device, use_mixup=True, mixup_alpha=1.0
    )
    results['组合增强'] = accs_combined
    
    return results


def visualize_augmented_samples():
    """可视化数据增强效果"""
    # 加载数据
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    
    # 获取一个样本
    original_img, label = dataset[0]
    
    # 应用不同增强
    elastic = ElasticDeformation(alpha=36, sigma=4)
    erasing = RandomErasing(probability=1.0)
    
    # 生成增强样本
    img_elastic = elastic(original_img.numpy())
    img_erasing = erasing(original_img.clone())
    
    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # 原始图像
    img = original_img[0].numpy() * 0.3081 + 0.1307
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title(f'原始图像\n标签: {label}', fontsize=12)
    axes[0].axis('off')
    
    # 弹性变形
    img_e = img_elastic[0] * 0.3081 + 0.1307
    axes[1].imshow(img_e, cmap='gray')
    axes[1].set_title('弹性变形', fontsize=12)
    axes[1].axis('off')
    
    # 随机擦除
    img_r = img_erasing[0].numpy() * 0.3081 + 0.1307
    axes[2].imshow(img_r, cmap='gray')
    axes[2].set_title('随机擦除', fontsize=12)
    axes[2].axis('off')
    
    plt.tight_layout()
    return fig


def main():
    """主函数"""
    print("\n" + "="*70)
    print("增量学习与数据增强实验")
    print("="*70)
    
    # 创建结果目录
    results_dir = 'results/incremental'
    os.makedirs(results_dir, exist_ok=True)
    
    # 实验1: 增量学习
    print("\n实验1: 测试EWC增量学习方法...")
    results_no_ewc, results_with_ewc = test_incremental_learning()
    
    # 可视化增量学习结果
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    categories = ['旧任务\n(训练前)', '旧任务\n(训练后)', '新任务', '遗忘率']
    no_ewc_values = [
        results_no_ewc['old_task_acc_before'],
        results_no_ewc['old_task_acc_after'],
        results_no_ewc['new_task_acc'],
        results_no_ewc['forgetting']
    ]
    with_ewc_values = [
        results_with_ewc['old_task_acc_before'],
        results_with_ewc['old_task_acc_after'],
        results_with_ewc['new_task_acc'],
        results_with_ewc['forgetting']
    ]
    
    x = np.arange(len(categories))
    width = 0.35
    
    axes[0].bar(x - width/2, no_ewc_values, width, label='无EWC', alpha=0.8)
    axes[0].bar(x + width/2, with_ewc_values, width, label='使用EWC', alpha=0.8)
    axes[0].set_ylabel('准确率/遗忘率 (%)', fontsize=11)
    axes[0].set_title('EWC增量学习效果对比', fontsize=13, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(categories, fontsize=10)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for i, (v1, v2) in enumerate(zip(no_ewc_values, with_ewc_values)):
        axes[0].text(i - width/2, v1 + 1, f'{v1:.1f}', ha='center', va='bottom', fontsize=9)
        axes[0].text(i + width/2, v2 + 1, f'{v2:.1f}', ha='center', va='bottom', fontsize=9)
    
    # 遗忘率对比
    methods = ['无EWC', '使用EWC']
    forgetting_rates = [results_no_ewc['forgetting'], results_with_ewc['forgetting']]
    colors = ['#d62728', '#2ca02c']
    
    axes[1].bar(methods, forgetting_rates, color=colors, alpha=0.8)
    axes[1].set_ylabel('遗忘率 (%)', fontsize=11)
    axes[1].set_title('遗忘率对比（越低越好）', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    for i, v in enumerate(forgetting_rates):
        axes[1].text(i, v + 0.5, f'{v:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'incremental_learning_comparison.png'), dpi=300)
    print(f"✓ 保存图表: {results_dir}/incremental_learning_comparison.png")
    plt.close()
    
    # 实验2: 数据增强
    print("\n实验2: 测试不同数据增强策略...")
    aug_results = test_data_augmentation()
    
    # 可视化数据增强结果
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for name, accs in aug_results.items():
        epochs = range(1, len(accs) + 1)
        ax.plot(epochs, accs, marker='o', label=name, linewidth=2)
    
    ax.set_xlabel('训练轮数 (Epoch)', fontsize=12)
    ax.set_ylabel('测试准确率 (%)', fontsize=12)
    ax.set_title('不同数据增强策略的效果对比', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'data_augmentation_comparison.png'), dpi=300)
    print(f"✓ 保存图表: {results_dir}/data_augmentation_comparison.png")
    plt.close()
    
    # 实验3: 可视化增强样本
    print("\n实验3: 可视化数据增强效果...")
    fig = visualize_augmented_samples()
    fig.savefig(os.path.join(results_dir, 'augmentation_examples.png'), dpi=300)
    print(f"✓ 保存图表: {results_dir}/augmentation_examples.png")
    plt.close()
    
    print(f"\n{'='*70}")
    print("✓ 增量学习与数据增强实验完成！")
    print(f"✓ 结果保存在: {results_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
