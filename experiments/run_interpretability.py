"""
可解释性分析实验

使用Grad-CAM生成注意力热力图，分析不同网络结构对图像关键区域的关注度
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib.pyplot as plt
import numpy as np

from src.main import get_data_loaders
from src.models import ConfigurableCNN
from src.interpretability import (GradCAM, analyze_layer_attention, 
                                 compare_architecture_attention, calculate_attention_metrics)
from src.train import run_training

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def visualize_gradcam_layers():
    """可视化不同层的Grad-CAM热力图"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载数据和训练模型
    train_loader, test_loader = get_data_loaders()
    model = ConfigurableCNN(num_conv_layers=3, kernel_size=3).to(device)
    
    print("\n训练模型...")
    run_training(model, train_loader, test_loader, epochs=5, device=device)
    
    # 分析不同层的注意力
    print("\n分析不同卷积层的注意力模式...")
    results = analyze_layer_attention(model, test_loader, device, num_samples=5)
    
    # 可视化
    num_samples = len(results['images'])
    num_layers = len(results['layer_names'])
    
    fig, axes = plt.subplots(num_samples, num_layers + 1, figsize=(3*(num_layers+1), 3*num_samples))
    
    for i in range(num_samples):
        # 原始图像
        img = results['images'][i][0, 0].cpu().numpy()
        img = img * 0.3081 + 0.1307
        axes[i, 0].imshow(img, cmap='gray')
        axes[i, 0].set_title(f'原始图像\n标签:{results["labels"][i]}', fontsize=10)
        axes[i, 0].axis('off')
        
        # 各层热力图
        for j in range(num_layers):
            heatmap = results['heatmaps'][j][i]
            axes[i, j+1].imshow(heatmap)
            axes[i, j+1].set_title(f'Layer {j+1}', fontsize=10)
            axes[i, j+1].axis('off')
    
    plt.tight_layout()
    return fig


def compare_architectures():
    """对比不同网络结构的注意力模式"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_loader, test_loader = get_data_loaders()
    
    # 训练不同架构的模型
    models_dict = {}
    
    configs = [
        (1, 3, "1层-3×3"),
        (2, 3, "2层-3×3"),
        (3, 3, "3层-3×3"),
        (2, 5, "2层-5×5"),
        (2, 7, "2层-7×7"),
    ]
    
    for num_layers, kernel_size, name in configs:
        print(f"\n训练模型: {name}")
        model = ConfigurableCNN(num_conv_layers=num_layers, kernel_size=kernel_size)
        run_training(model, train_loader, test_loader, epochs=5, device=device)
        models_dict[name] = model
    
    # 对比分析
    print("\n对比不同架构的注意力模式...")
    comparison = compare_architecture_attention(models_dict, test_loader, device, num_samples=5)
    
    # 可视化
    num_samples = len(comparison['images'])
    num_models = len(models_dict)
    
    fig, axes = plt.subplots(num_samples, num_models + 1, figsize=(3*(num_models+1), 3*num_samples))
    
    for i in range(num_samples):
        # 原始图像
        img = comparison['images'][i][0, 0].cpu().numpy()
        img = img * 0.3081 + 0.1307
        axes[i, 0].imshow(img, cmap='gray')
        axes[i, 0].set_title(f'原始\n标签:{comparison["labels"][i]}', fontsize=9)
        axes[i, 0].axis('off')
        
        # 各模型热力图
        for j, (model_name, data) in enumerate(comparison['models'].items()):
            heatmap = data['heatmaps'][i]
            pred = data['predictions'][i]
            axes[i, j+1].imshow(heatmap)
            axes[i, j+1].set_title(f'{model_name}\n预测:{pred}', fontsize=9)
            axes[i, j+1].axis('off')
    
    plt.tight_layout()
    return fig, comparison


def analyze_attention_patterns():
    """分析注意力模式的统计规律"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_loader, test_loader = get_data_loaders()
    
    # 测试不同配置
    configs = [
        (1, 3, "1层"),
        (2, 3, "2层"),
        (3, 3, "3层"),
    ]
    
    results = {}
    
    for num_layers, kernel_size, name in configs:
        print(f"\n分析 {name} 网络...")
        model = ConfigurableCNN(num_conv_layers=num_layers, kernel_size=kernel_size).to(device)
        run_training(model, train_loader, test_loader, epochs=5, device=device)
        
        # 找到最后一个卷积层
        last_conv = None
        for module in model.modules():
            if isinstance(module, torch.nn.Conv2d):
                last_conv = module
        
        # 生成热力图并计算指标
        gradcam = GradCAM(model, last_conv)
        metrics_list = []
        
        for images, labels in test_loader:
            for i in range(min(10, images.size(0))):
                img = images[i:i+1].to(device)
                heatmap = gradcam.generate_heatmap(img)
                metrics = calculate_attention_metrics(heatmap)
                metrics_list.append(metrics)
            break
        
        # 计算平均指标
        avg_metrics = {}
        for key in metrics_list[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in metrics_list])
        
        results[name] = avg_metrics
        
        print(f"  平均注意力: {avg_metrics['mean_attention']:.3f}")
        print(f"  中心注意力: {avg_metrics['center_attention']:.3f}")
        print(f"  边缘注意力: {avg_metrics['edge_attention']:.3f}")
    
    return results


def main():
    """主函数"""
    print("\n" + "="*70)
    print("可解释性分析实验")
    print("="*70)
    
    # 创建结果目录
    results_dir = 'results/interpretability'
    os.makedirs(results_dir, exist_ok=True)
    
    # 实验1: 不同层的热力图
    print("\n实验1: 可视化不同卷积层的注意力...")
    fig1 = visualize_gradcam_layers()
    fig1.savefig(os.path.join(results_dir, 'layer_attention.png'), dpi=300, bbox_inches='tight')
    print(f"✓ 保存图表: {results_dir}/layer_attention.png")
    plt.close()
    
    # 实验2: 对比不同架构
    print("\n实验2: 对比不同网络结构的注意力模式...")
    fig2, comparison = compare_architectures()
    fig2.savefig(os.path.join(results_dir, 'architecture_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"✓ 保存图表: {results_dir}/architecture_comparison.png")
    plt.close()
    
    # 实验3: 统计分析
    print("\n实验3: 分析注意力模式的统计规律...")
    stats = analyze_attention_patterns()
    
    # 可视化统计结果
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    names = list(stats.keys())
    metrics = ['mean_attention', 'center_attention', 'edge_attention']
    titles = ['平均注意力', '中心区域注意力', '边缘区域注意力']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        values = [stats[name][metric] for name in names]
        axes[idx].bar(names, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        axes[idx].set_ylabel('注意力强度', fontsize=11)
        axes[idx].set_title(title, fontsize=12, fontweight='bold')
        axes[idx].grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for i, v in enumerate(values):
            axes[idx].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'attention_statistics.png'), dpi=300)
    print(f"✓ 保存图表: {results_dir}/attention_statistics.png")
    plt.close()
    
    print(f"\n{'='*70}")
    print("✓ 可解释性分析实验完成！")
    print(f"✓ 结果保存在: {results_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
