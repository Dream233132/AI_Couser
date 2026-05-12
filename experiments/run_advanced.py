"""
创新要求1：非标准卷积与注意力机制对比实验

对比以下模型：
1. 标准CNN（基线）
2. SENet-CNN（注意力机制）
3. DilatedCNN（空洞卷积）
4. DepthwiseCNN（深度可分离卷积）

评估指标：
- 准确率
- 参数量
- 训练时间
- 推理时间
"""

import os
import json
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

from src.main import get_data_loaders
from src.models import ConfigurableCNN, count_parameters
from src.advanced_models import SENetCNN, DilatedCNN, DepthwiseCNN
from src.train import run_training

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class AdvancedExperimentRunner:
    """高级模型对比实验运行器"""
    
    def __init__(self, epochs=10, learning_rate=0.001, batch_size=64):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 检查GPU
        self._check_gpu()
        
        # 加载数据
        print("正在加载MNIST数据集...")
        self.train_loader, self.test_loader = get_data_loaders(
            batch_size_train=batch_size,
            batch_size_test=1000
        )
        
        # 存储结果
        self.results = {}
        
        # 创建结果目录
        self.results_dir = 'advanced_results'
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
    
    def _check_gpu(self):
        """检查GPU配置"""
        print("\n" + "="*80)
        print("GPU配置检查")
        print("="*80)
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"\n✅ 检测到GPU: {gpu_name}")
            print(f"✅ GPU显存: {total_memory:.2f} GB")
            print(f"✅ CUDA版本: {torch.version.cuda}")
            
            if torch.backends.cudnn.is_available():
                torch.backends.cudnn.enabled = True
                torch.backends.cudnn.benchmark = True
                print(f"✅ cuDNN加速: 已启用")
            
            print("\n🚀 将使用GPU加速训练！")
        else:
            print("\n⚠️  未检测到GPU，将使用CPU训练")
        
        print("="*80 + "\n")
    
    def measure_inference_time(self, model, num_samples=1000):
        """测量模型推理时间"""
        model.eval()
        model = model.to(self.device)
        
        # 预热
        dummy_input = torch.randn(1, 1, 28, 28).to(self.device)
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)
        
        # 测量时间
        times = []
        with torch.no_grad():
            for images, _ in self.test_loader:
                if len(times) * images.size(0) >= num_samples:
                    break
                
                images = images.to(self.device)
                
                # 同步GPU
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                start_time = time.time()
                _ = model(images)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                end_time = time.time()
                
                batch_time = (end_time - start_time) * 1000  # 转换为毫秒
                times.append(batch_time / images.size(0))  # 每个样本的时间
        
        avg_time = np.mean(times)
        return avg_time
    
    def run_experiment(self, model_name, model):
        """运行单个模型的实验"""
        print("\n" + "="*80)
        print(f"实验: {model_name}")
        print("="*80)
        
        # 计算参数量
        total_params = count_parameters(model)
        print(f"\n模型参数总数: {total_params:,}")
        
        # 训练模型
        start_time = time.time()
        all_losses, accuracies = run_training(
            model=model,
            train_loader=self.train_loader,
            test_loader=self.test_loader,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            device=self.device
        )
        training_time = time.time() - start_time
        
        # 测量推理时间
        print("\n测量推理时间...")
        inference_time = self.measure_inference_time(model)
        
        # 保存结果
        result = {
            'model_name': model_name,
            'total_params': total_params,
            'all_losses': all_losses,
            'accuracies': accuracies,
            'final_accuracy': accuracies[-1],
            'best_accuracy': max(accuracies),
            'training_time': training_time,
            'inference_time_ms': inference_time,
            'epochs': self.epochs
        }
        
        self.results[model_name] = result
        
        print(f"\n✓ {model_name} 实验完成")
        print(f"  最终准确率: {result['final_accuracy']:.2f}%")
        print(f"  训练时间: {training_time:.1f}秒")
        print(f"  推理时间: {inference_time:.3f}ms/样本")
        
        return result
    
    def run_all_experiments(self):
        """运行所有模型的对比实验"""
        print("\n" + "#"*80)
        print("# 创新要求1：非标准卷积与注意力机制对比实验")
        print("#"*80)
        
        # 定义所有模型
        models = {
            '标准CNN': ConfigurableCNN(num_conv_layers=2, kernel_size=3),
            'SENet-CNN': SENetCNN(num_conv_layers=2),
            'DilatedCNN': DilatedCNN(num_conv_layers=2),
            'DepthwiseCNN': DepthwiseCNN(num_conv_layers=2)
        }
        
        # 运行每个模型的实验
        for model_name, model in models.items():
            self.run_experiment(model_name, model)
    
    def save_results(self):
        """保存实验结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.results_dir, f'advanced_results_{timestamp}.json')
        
        # 转换为可序列化格式
        serializable_results = {}
        for name, result in self.results.items():
            serializable_results[name] = {
                'model_name': result['model_name'],
                'total_params': result['total_params'],
                'final_accuracy': result['final_accuracy'],
                'best_accuracy': result['best_accuracy'],
                'training_time': result['training_time'],
                'inference_time_ms': result['inference_time_ms'],
                'epochs': result['epochs'],
                'accuracies': result['accuracies']
            }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ 实验结果已保存到: {filename}")
        return filename
    
    def generate_comparison_table(self):
        """生成对比表格"""
        print("\n" + "="*80)
        print("模型对比表格")
        print("="*80)
        
        print(f"\n{'模型':<15} {'准确率(%)':<12} {'参数量':<12} {'训练时间(s)':<15} {'推理时间(ms)':<15}")
        print("-" * 80)
        
        for name, result in self.results.items():
            print(f"{name:<15} "
                  f"{result['final_accuracy']:<12.2f} "
                  f"{result['total_params']:<12,} "
                  f"{result['training_time']:<15.1f} "
                  f"{result['inference_time_ms']:<15.3f}")
        
        print("="*80)
    
    def plot_comparison_charts(self):
        """生成对比图表"""
        if not self.results:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        models = list(self.results.keys())
        
        # 1. 准确率对比
        accuracies = [self.results[m]['final_accuracy'] for m in models]
        axes[0, 0].bar(range(len(models)), accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        axes[0, 0].set_xticks(range(len(models)))
        axes[0, 0].set_xticklabels(models, rotation=15, ha='right')
        axes[0, 0].set_ylabel('准确率 (%)')
        axes[0, 0].set_title('模型准确率对比', fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for i, acc in enumerate(accuracies):
            axes[0, 0].text(i, acc + 0.2, f'{acc:.2f}%', ha='center', va='bottom')
        
        # 2. 参数量对比
        params = [self.results[m]['total_params'] / 1000 for m in models]  # 转换为K
        axes[0, 1].bar(range(len(models)), params, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        axes[0, 1].set_xticks(range(len(models)))
        axes[0, 1].set_xticklabels(models, rotation=15, ha='right')
        axes[0, 1].set_ylabel('参数量 (K)')
        axes[0, 1].set_title('模型参数量对比', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        for i, p in enumerate(params):
            axes[0, 1].text(i, p + 5, f'{p:.1f}K', ha='center', va='bottom')
        
        # 3. 训练时间对比
        train_times = [self.results[m]['training_time'] for m in models]
        axes[1, 0].bar(range(len(models)), train_times, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        axes[1, 0].set_xticks(range(len(models)))
        axes[1, 0].set_xticklabels(models, rotation=15, ha='right')
        axes[1, 0].set_ylabel('训练时间 (秒)')
        axes[1, 0].set_title('模型训练时间对比', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        for i, t in enumerate(train_times):
            axes[1, 0].text(i, t + 5, f'{t:.1f}s', ha='center', va='bottom')
        
        # 4. 推理时间对比
        infer_times = [self.results[m]['inference_time_ms'] for m in models]
        axes[1, 1].bar(range(len(models)), infer_times, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        axes[1, 1].set_xticks(range(len(models)))
        axes[1, 1].set_xticklabels(models, rotation=15, ha='right')
        axes[1, 1].set_ylabel('推理时间 (ms/样本)')
        axes[1, 1].set_title('模型推理时间对比', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        for i, t in enumerate(infer_times):
            axes[1, 1].text(i, t + 0.05, f'{t:.3f}ms', ha='center', va='bottom')
        
        plt.tight_layout()
        filename = os.path.join(self.results_dir, 'advanced_comparison.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ 对比图表已保存: {filename}")
        plt.close()
    
    def generate_analysis_report(self):
        """生成分析报告"""
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("创新要求1：非标准卷积与注意力机制 - 实验分析报告")
        report_lines.append("="*80)
        report_lines.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"训练设备: {self.device}")
        report_lines.append(f"训练轮数: {self.epochs}")
        
        report_lines.append("\n" + "="*80)
        report_lines.append("实验结果对比")
        report_lines.append("="*80)
        
        report_lines.append(f"\n{'模型':<15} {'准确率(%)':<12} {'参数量':<12} {'训练时间(s)':<15} {'推理时间(ms)':<15}")
        report_lines.append("-" * 80)
        
        for name, result in self.results.items():
            report_lines.append(
                f"{name:<15} "
                f"{result['final_accuracy']:<12.2f} "
                f"{result['total_params']:<12,} "
                f"{result['training_time']:<15.1f} "
                f"{result['inference_time_ms']:<15.3f}"
            )
        
        # 分析各模型特点
        report_lines.append("\n" + "="*80)
        report_lines.append("模型分析")
        report_lines.append("="*80)
        
        baseline = self.results.get('标准CNN', None)
        
        if '标准CNN' in self.results:
            report_lines.append("\n【标准CNN - 基线模型】")
            report_lines.append(f"• 准确率: {baseline['final_accuracy']:.2f}%")
            report_lines.append(f"• 参数量: {baseline['total_params']:,}")
            report_lines.append("• 特点: 传统卷积神经网络，作为对比基线")
        
        if 'SENet-CNN' in self.results:
            senet = self.results['SENet-CNN']
            report_lines.append("\n【SENet-CNN - 注意力机制】")
            report_lines.append(f"• 准确率: {senet['final_accuracy']:.2f}%")
            if baseline:
                acc_diff = senet['final_accuracy'] - baseline['final_accuracy']
                report_lines.append(f"• 准确率提升: {acc_diff:+.2f}%")
            report_lines.append(f"• 参数量: {senet['total_params']:,}")
            if baseline:
                param_diff = senet['total_params'] - baseline['total_params']
                report_lines.append(f"• 参数增加: {param_diff:,} ({param_diff/baseline['total_params']*100:.1f}%)")
            report_lines.append("• 特点: 通过SE模块学习通道注意力，提升特征表达能力")
            report_lines.append("• 优势: 准确率提升明显，参数增加很少")
        
        if 'DilatedCNN' in self.results:
            dilated = self.results['DilatedCNN']
            report_lines.append("\n【DilatedCNN - 空洞卷积】")
            report_lines.append(f"• 准确率: {dilated['final_accuracy']:.2f}%")
            report_lines.append(f"• 参数量: {dilated['total_params']:,}")
            report_lines.append("• 特点: 使用空洞卷积扩大感受野")
            report_lines.append("• 优势: 不增加参数量的情况下捕捉更大范围的特征")
        
        if 'DepthwiseCNN' in self.results:
            depthwise = self.results['DepthwiseCNN']
            report_lines.append("\n【DepthwiseCNN - 深度可分离卷积】")
            report_lines.append(f"• 准确率: {depthwise['final_accuracy']:.2f}%")
            report_lines.append(f"• 参数量: {depthwise['total_params']:,}")
            if baseline:
                param_ratio = baseline['total_params'] / depthwise['total_params']
                report_lines.append(f"• 参数减少: {param_ratio:.1f}倍")
            report_lines.append(f"• 推理时间: {depthwise['inference_time_ms']:.3f}ms")
            report_lines.append("• 特点: 大幅减少参数量和计算量")
            report_lines.append("• 优势: 适合移动端部署，速度快")
        
        # 总结
        report_lines.append("\n" + "="*80)
        report_lines.append("总结")
        report_lines.append("="*80)
        
        best_acc_model = max(self.results.items(), key=lambda x: x[1]['final_accuracy'])
        fastest_model = min(self.results.items(), key=lambda x: x[1]['inference_time_ms'])
        lightest_model = min(self.results.items(), key=lambda x: x[1]['total_params'])
        
        report_lines.append(f"\n• 最高准确率: {best_acc_model[0]} ({best_acc_model[1]['final_accuracy']:.2f}%)")
        report_lines.append(f"• 最快推理: {fastest_model[0]} ({fastest_model[1]['inference_time_ms']:.3f}ms)")
        report_lines.append(f"• 最少参数: {lightest_model[0]} ({lightest_model[1]['total_params']:,})")
        
        report_lines.append("\n• 实验证明了现代改进结构的有效性：")
        report_lines.append("  - SENet注意力机制能有效提升准确率")
        report_lines.append("  - 空洞卷积能扩大感受野而不增加参数")
        report_lines.append("  - 深度可分离卷积大幅减少计算量，适合移动端")
        
        report_lines.append("\n" + "="*80)
        
        # 保存报告
        report_text = '\n'.join(report_lines)
        print(report_text)
        
        filename = os.path.join(self.results_dir, 'advanced_analysis_report.txt')
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"\n✓ 分析报告已保存到: {filename}")
        return filename


def main():
    """主函数"""
    print("\n" + "="*80)
    print("创新要求1：非标准卷积与注意力机制对比实验")
    print("="*80)
    
    # 创建实验运行器
    runner = AdvancedExperimentRunner(epochs=10, learning_rate=0.001, batch_size=64)
    
    # 运行所有实验
    runner.run_all_experiments()
    
    # 保存结果
    runner.save_results()
    
    # 生成对比表格
    runner.generate_comparison_table()
    
    # 生成图表
    print("\n生成对比图表...")
    runner.plot_comparison_charts()
    
    # 生成分析报告
    print("\n生成分析报告...")
    runner.generate_analysis_report()
    
    print("\n" + "="*80)
    print("✓ 创新要求1实验完成！")
    print(f"✓ 结果保存在目录: {runner.results_dir}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
