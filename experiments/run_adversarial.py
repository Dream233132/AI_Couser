"""
对抗鲁棒性测试实验

测试不同网络结构在对抗攻击下的鲁棒性，并实现对抗训练防御
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import json

from src.main import get_data_loaders
from src.models import ConfigurableCNN, count_parameters
from src.adversarial import (FGSMAttack, PGDAttack, evaluate_robustness,
                             adversarial_training, generate_adversarial_examples)
from src.train import run_training

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def test_robustness_vs_architecture():
    """测试不同网络结构的鲁棒性"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}\n")
    
    # 加载数据
    _, test_loader = get_data_loaders(batch_size_test=100)
    
    # 测试配置
    architectures = [
        (1, 3, "1层-3×3"),
        (2, 3, "2层-3×3"),
        (3, 3, "3层-3×3"),
        (2, 5, "2层-5×5"),
        (2, 7, "2层-7×7"),
    ]
    
    epsilons = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]
    
    results = {}
    
    for num_layers, kernel_size, name in architectures:
        print(f"\n{'='*70}")
        print(f"测试架构: {name}")
        print(f"{'='*70}")
        
        # 创建并训练模型
        model = ConfigurableCNN(num_conv_layers=num_layers, kernel_size=kernel_size)
        model = model.to(device)
        
        # 简单训练（5个epoch）
        train_loader, _ = get_data_loaders(batch_size_train=64)
        print("\n训练模型...")
        run_training(model, train_loader, test_loader, epochs=5, 
                    learning_rate=0.001, device=device)
        
        # 测试不同epsilon下的鲁棒性
        arch_results = {'epsilons': epsilons, 'accuracies': []}
        
        for eps in epsilons:
            if eps == 0.0:
                # 干净样本准确率
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
                acc = 100.0 * correct / total
            else:
                # 对抗样本准确率
                attack = FGSMAttack(epsilon=eps)
                _, acc = evaluate_robustness(model, test_loader, attack, device)
            
            arch_results['accuracies'].append(acc)
            print(f"  ε={eps:.2f}: {acc:.2f}%")
        
        results[name] = arch_results
    
    return results


def visualize_adversarial_examples():
    """可视化对抗样本"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载数据和模型
    train_loader, test_loader = get_data_loaders()
    model = ConfigurableCNN(num_conv_layers=2, kernel_size=3).to(device)
    
    # 训练模型
    print("\n训练模型用于可视化...")
    run_training(model, train_loader, test_loader, epochs=5, device=device)
    
    # 生成对抗样本
    epsilon = 0.3
    orig_imgs, adv_imgs, labels, orig_preds, adv_preds = generate_adversarial_examples(
        model, test_loader, epsilon, device, num_examples=10
    )
    
    # 可视化
    fig, axes = plt.subplots(3, 10, figsize=(20, 6))
    
    for i in range(10):
        # 原始图像
        img = orig_imgs[i, 0].numpy()
        img = img * 0.3081 + 0.1307
        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].set_title(f'原始\n真实:{labels[i]}\n预测:{orig_preds[i]}', fontsize=8)
        axes[0, i].axis('off')
        
        # 对抗样本
        adv_img = adv_imgs[i, 0].numpy()
        adv_img = adv_img * 0.3081 + 0.1307
        axes[1, i].imshow(adv_img, cmap='gray')
        axes[1, i].set_title(f'对抗\n预测:{adv_preds[i]}', fontsize=8)
        axes[1, i].axis('off')
        
        # 扰动
        perturbation = np.abs(adv_img - img)
        axes[2, i].imshow(perturbation, cmap='hot')
        axes[2, i].set_title(f'扰动\nε={epsilon}', fontsize=8)
        axes[2, i].axis('off')
    
    plt.tight_layout()
    
    return fig


def test_adversarial_training():
    """测试对抗训练防御效果"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_loader, test_loader = get_data_loaders()
    
    # 1. 标准训练
    print("\n1. 标准训练模型...")
    model_standard = ConfigurableCNN(num_conv_layers=2, kernel_size=3)
    run_training(model_standard, train_loader, test_loader, 
                epochs=5, device=device)
    
    # 2. 对抗训练
    print("\n2. 对抗训练模型...")
    model_robust = ConfigurableCNN(num_conv_layers=2, kernel_size=3)
    clean_accs, adv_accs = adversarial_training(
        model_robust, train_loader, test_loader,
        epochs=5, learning_rate=0.001, epsilon=0.1, device=device
    )
    
    # 3. 对比测试
    print("\n3. 对比测试...")
    epsilons = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]
    
    standard_accs = []
    robust_accs = []
    
    for eps in epsilons:
        if eps == 0.0:
            # 干净样本
            attack = FGSMAttack(epsilon=0.01)  # 占位
            clean_acc_std, _ = evaluate_robustness(model_standard, test_loader, attack, device)
            clean_acc_rob, _ = evaluate_robustness(model_robust, test_loader, attack, device)
            standard_accs.append(clean_acc_std)
            robust_accs.append(clean_acc_rob)
        else:
            attack = FGSMAttack(epsilon=eps)
            _, adv_acc_std = evaluate_robustness(model_standard, test_loader, attack, device)
            _, adv_acc_rob = evaluate_robustness(model_robust, test_loader, attack, device)
            standard_accs.append(adv_acc_std)
            robust_accs.append(adv_acc_rob)
        
        print(f"  ε={eps:.2f} - 标准: {standard_accs[-1]:.2f}%, 对抗训练: {robust_accs[-1]:.2f}%")
    
    return epsilons, standard_accs, robust_accs


def main():
    """主函数"""
    print("\n" + "="*70)
    print("对抗鲁棒性测试实验")
    print("="*70)
    
    # 创建结果目录
    results_dir = 'results/adversarial'
    os.makedirs(results_dir, exist_ok=True)
    
    # 实验1: 不同架构的鲁棒性
    print("\n实验1: 测试不同网络结构的鲁棒性...")
    arch_results = test_robustness_vs_architecture()
    
    # 绘制结果
    plt.figure(figsize=(12, 6))
    for name, data in arch_results.items():
        plt.plot(data['epsilons'], data['accuracies'], marker='o', label=name, linewidth=2)
    
    plt.xlabel('扰动强度 ε', fontsize=12)
    plt.ylabel('准确率 (%)', fontsize=12)
    plt.title('不同网络结构在对抗攻击下的鲁棒性', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'robustness_vs_architecture.png'), dpi=300)
    print(f"✓ 保存图表: {results_dir}/robustness_vs_architecture.png")
    plt.close()
    
    # 实验2: 可视化对抗样本
    print("\n实验2: 生成对抗样本可视化...")
    fig = visualize_adversarial_examples()
    fig.savefig(os.path.join(results_dir, 'adversarial_examples.png'), dpi=300)
    print(f"✓ 保存图表: {results_dir}/adversarial_examples.png")
    plt.close()
    
    # 实验3: 对抗训练防御
    print("\n实验3: 测试对抗训练防御效果...")
    epsilons, standard_accs, robust_accs = test_adversarial_training()
    
    # 绘制对比
    plt.figure(figsize=(10, 6))
    plt.plot(epsilons, standard_accs, marker='o', label='标准训练', linewidth=2)
    plt.plot(epsilons, robust_accs, marker='s', label='对抗训练', linewidth=2)
    plt.xlabel('扰动强度 ε', fontsize=12)
    plt.ylabel('准确率 (%)', fontsize=12)
    plt.title('对抗训练防御效果', fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'adversarial_training_defense.png'), dpi=300)
    print(f"✓ 保存图表: {results_dir}/adversarial_training_defense.png")
    plt.close()
    
    # 保存结果
    results = {
        'architecture_robustness': arch_results,
        'defense_comparison': {
            'epsilons': epsilons,
            'standard_training': standard_accs,
            'adversarial_training': robust_accs
        }
    }
    
    with open(os.path.join(results_dir, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*70}")
    print("✓ 对抗鲁棒性测试实验完成！")
    print(f"✓ 结果保存在: {results_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
