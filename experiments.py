"""
基于卷积神经网络的MNIST数据分类系统 - 完整实验脚本

本脚本实现课设的所有基础要求：
A) 训练网络并记录每次迭代的损失值
B) 改变卷积层和池化层数，观察分类准确率
C) 改变卷积核大小，观察分类准确率
D) 分析改进的卷积神经网络数据分类的有效性
E) 设计实现基于卷积神经网络的数据分类系统
"""

import os
import json
import torch
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from src.main import get_data_loaders
from src.models import ConfigurableCNN, count_parameters
from src.train import run_training, print_training_summary

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class ExperimentRunner:
    """实验运行器 - 管理所有对比实验"""
    
    def __init__(self, epochs=10, learning_rate=0.001, batch_size=64):
        """
        初始化实验运行器
        
        参数：
            epochs (int): 训练轮数
            learning_rate (float): 学习率
            batch_size (int): 批大小
        """
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        # 检查并显示GPU信息
        self._check_gpu()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载数据
        print("正在加载MNIST数据集...")
        self.train_loader, self.test_loader = get_data_loaders(
            batch_size_train=batch_size,
            batch_size_test=1000
        )
        
        # 存储实验结果
        self.results = {}
        
        # 创建结果目录
        self.results_dir = 'experiment_results'
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
    
    def _check_gpu(self):
        """检查GPU配置并显示信息"""
        print("\n" + "="*80)
        print("GPU配置检查")
        print("="*80)
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"\n✅ 检测到GPU: {gpu_name}")
            print(f"✅ GPU显存: {total_memory:.2f} GB")
            print(f"✅ CUDA版本: {torch.version.cuda}")
            
            # 启用cuDNN加速
            if torch.backends.cudnn.is_available():
                torch.backends.cudnn.enabled = True
                torch.backends.cudnn.benchmark = True
                print(f"✅ cuDNN加速: 已启用")
            
            print("\n🚀 将使用GPU加速训练！")
        else:
            print("\n⚠️  未检测到GPU，将使用CPU训练")
            print("提示: 安装CUDA版本的PyTorch可以大幅提升训练速度")
        
        print("="*80 + "\n")
    
    def run_single_experiment(self, num_conv_layers, kernel_size, experiment_name):
        """
        运行单个实验
        
        参数：
            num_conv_layers (int): 卷积层数
            kernel_size (int): 卷积核大小
            experiment_name (str): 实验名称
        
        返回：
            dict: 实验结果
        """
        print("\n" + "="*80)
        print(f"实验: {experiment_name}")
        print(f"配置: {num_conv_layers}层卷积, {kernel_size}×{kernel_size}卷积核")
        print("="*80)
        
        # 创建模型
        model = ConfigurableCNN(
            num_conv_layers=num_conv_layers,
            kernel_size=kernel_size
        )
        
        # 计算参数数量
        total_params = count_parameters(model)
        print(f"\n模型参数总数: {total_params:,}")
        
        # 训练模型
        all_losses, accuracies = run_training(
            model=model,
            train_loader=self.train_loader,
            test_loader=self.test_loader,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            device=self.device
        )
        
        # 保存结果
        result = {
            'experiment_name': experiment_name,
            'num_conv_layers': num_conv_layers,
            'kernel_size': kernel_size,
            'total_params': total_params,
            'all_losses': all_losses,
            'accuracies': accuracies,
            'final_accuracy': accuracies[-1],
            'best_accuracy': max(accuracies),
            'epochs': self.epochs,
            'learning_rate': self.learning_rate
        }
        
        self.results[experiment_name] = result
        
        return result
    
    def experiment_b_layer_comparison(self):
        """
        实验B: 改变卷积层和池化层数，观察分类准确率
        
        测试配置：
        - 1层卷积 + 1层池化
        - 2层卷积 + 2层池化
        - 3层卷积 + 3层池化
        """
        print("\n" + "#"*80)
        print("# 实验B: 网络层数对分类准确率的影响")
        print("#"*80)
        
        layer_configs = [1, 2, 3]
        kernel_size = 3  # 固定卷积核大小为3×3
        
        for num_layers in layer_configs:
            experiment_name = f"B_layers_{num_layers}"
            self.run_single_experiment(
                num_conv_layers=num_layers,
                kernel_size=kernel_size,
                experiment_name=experiment_name
            )
    
    def experiment_c_kernel_comparison(self):
        """
        实验C: 改变卷积核大小，观察分类准确率
        
        测试配置：
        - 3×3 卷积核
        - 5×5 卷积核
        - 7×7 卷积核
        """
        print("\n" + "#"*80)
        print("# 实验C: 卷积核大小对分类准确率的影响")
        print("#"*80)
        
        kernel_configs = [3, 5, 7]
        num_layers = 2  # 固定层数为2
        
        for kernel_sz in kernel_configs:
            experiment_name = f"C_kernel_{kernel_sz}"
            self.run_single_experiment(
                num_conv_layers=num_layers,
                kernel_size=kernel_sz,
                experiment_name=experiment_name
            )
    
    def save_results(self):
        """保存实验结果到JSON文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.results_dir, f'results_{timestamp}.json')
        
        # 转换为可序列化的格式
        serializable_results = {}
        for name, result in self.results.items():
            serializable_results[name] = {
                'experiment_name': result['experiment_name'],
                'num_conv_layers': result['num_conv_layers'],
                'kernel_size': result['kernel_size'],
                'total_params': result['total_params'],
                'final_accuracy': result['final_accuracy'],
                'best_accuracy': result['best_accuracy'],
                'epochs': result['epochs'],
                'learning_rate': result['learning_rate'],
                'accuracies': result['accuracies']
            }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ 实验结果已保存到: {filename}")
        return filename
    
    def plot_loss_curves(self):
        """绘制损失曲线 - 要求A"""
        if not self.results:
            print("没有实验结果可绘制")
            return
        
        # 为每个实验组创建图表
        for group_name in ['B', 'C']:
            group_results = {k: v for k, v in self.results.items() if k.startswith(group_name)}
            
            if not group_results:
                continue
            
            plt.figure(figsize=(12, 6))
            
            for exp_name, result in group_results.items():
                losses = result['all_losses']
                # 平滑处理
                window_size = 50
                if len(losses) > window_size:
                    smoothed = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
                else:
                    smoothed = losses
                
                label = f"{result['num_conv_layers']}层" if group_name == 'B' else f"{result['kernel_size']}×{result['kernel_size']}"
                plt.plot(smoothed, label=label, linewidth=2)
            
            plt.xlabel('迭代次数 (Iteration)', fontsize=12)
            plt.ylabel('损失值 (Loss)', fontsize=12)
            title = '网络层数对训练损失的影响' if group_name == 'B' else '卷积核大小对训练损失的影响'
            plt.title(title, fontsize=14, fontweight='bold')
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            filename = os.path.join(self.results_dir, f'loss_curve_{group_name}.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"✓ 损失曲线已保存: {filename}")
            plt.close()
    
    def plot_accuracy_comparison(self):
        """绘制准确率对比图 - 要求B和C"""
        if not self.results:
            print("没有实验结果可绘制")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 实验B: 层数对比
        b_results = {k: v for k, v in self.results.items() if k.startswith('B_')}
        if b_results:
            layers = []
            final_accs = []
            best_accs = []
            
            for result in sorted(b_results.values(), key=lambda x: x['num_conv_layers']):
                layers.append(f"{result['num_conv_layers']}层")
                final_accs.append(result['final_accuracy'])
                best_accs.append(result['best_accuracy'])
            
            x = np.arange(len(layers))
            width = 0.35
            
            axes[0].bar(x - width/2, final_accs, width, label='最终准确率', alpha=0.8)
            axes[0].bar(x + width/2, best_accs, width, label='最佳准确率', alpha=0.8)
            axes[0].set_xlabel('网络层数', fontsize=12)
            axes[0].set_ylabel('准确率 (%)', fontsize=12)
            axes[0].set_title('网络层数对分类准确率的影响', fontsize=14, fontweight='bold')
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(layers)
            axes[0].legend(fontsize=10)
            axes[0].grid(True, alpha=0.3, axis='y')
            
            # 添加数值标签
            for i, (f, b) in enumerate(zip(final_accs, best_accs)):
                axes[0].text(i - width/2, f + 0.5, f'{f:.2f}', ha='center', va='bottom', fontsize=9)
                axes[0].text(i + width/2, b + 0.5, f'{b:.2f}', ha='center', va='bottom', fontsize=9)
        
        # 实验C: 卷积核大小对比
        c_results = {k: v for k, v in self.results.items() if k.startswith('C_')}
        if c_results:
            kernels = []
            final_accs = []
            best_accs = []
            
            for result in sorted(c_results.values(), key=lambda x: x['kernel_size']):
                kernels.append(f"{result['kernel_size']}×{result['kernel_size']}")
                final_accs.append(result['final_accuracy'])
                best_accs.append(result['best_accuracy'])
            
            x = np.arange(len(kernels))
            width = 0.35
            
            axes[1].bar(x - width/2, final_accs, width, label='最终准确率', alpha=0.8)
            axes[1].bar(x + width/2, best_accs, width, label='最佳准确率', alpha=0.8)
            axes[1].set_xlabel('卷积核大小', fontsize=12)
            axes[1].set_ylabel('准确率 (%)', fontsize=12)
            axes[1].set_title('卷积核大小对分类准确率的影响', fontsize=14, fontweight='bold')
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(kernels)
            axes[1].legend(fontsize=10)
            axes[1].grid(True, alpha=0.3, axis='y')
            
            # 添加数值标签
            for i, (f, b) in enumerate(zip(final_accs, best_accs)):
                axes[1].text(i - width/2, f + 0.5, f'{f:.2f}', ha='center', va='bottom', fontsize=9)
                axes[1].text(i + width/2, b + 0.5, f'{b:.2f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        filename = os.path.join(self.results_dir, 'accuracy_comparison.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ 准确率对比图已保存: {filename}")
        plt.close()
    
    def plot_accuracy_curves(self):
        """绘制训练过程中的准确率变化曲线"""
        if not self.results:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 实验B
        b_results = {k: v for k, v in self.results.items() if k.startswith('B_')}
        if b_results:
            for result in sorted(b_results.values(), key=lambda x: x['num_conv_layers']):
                epochs = range(1, len(result['accuracies']) + 1)
                label = f"{result['num_conv_layers']}层卷积"
                axes[0].plot(epochs, result['accuracies'], marker='o', label=label, linewidth=2)
            
            axes[0].set_xlabel('训练轮数 (Epoch)', fontsize=12)
            axes[0].set_ylabel('测试准确率 (%)', fontsize=12)
            axes[0].set_title('网络层数 - 准确率变化曲线', fontsize=14, fontweight='bold')
            axes[0].legend(fontsize=10)
            axes[0].grid(True, alpha=0.3)
        
        # 实验C
        c_results = {k: v for k, v in self.results.items() if k.startswith('C_')}
        if c_results:
            for result in sorted(c_results.values(), key=lambda x: x['kernel_size']):
                epochs = range(1, len(result['accuracies']) + 1)
                label = f"{result['kernel_size']}×{result['kernel_size']}卷积核"
                axes[1].plot(epochs, result['accuracies'], marker='s', label=label, linewidth=2)
            
            axes[1].set_xlabel('训练轮数 (Epoch)', fontsize=12)
            axes[1].set_ylabel('测试准确率 (%)', fontsize=12)
            axes[1].set_title('卷积核大小 - 准确率变化曲线', fontsize=14, fontweight='bold')
            axes[1].legend(fontsize=10)
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = os.path.join(self.results_dir, 'accuracy_curves.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ 准确率变化曲线已保存: {filename}")
        plt.close()
    
    def generate_analysis_report(self):
        """生成分析报告 - 要求D"""
        if not self.results:
            print("没有实验结果可分析")
            return
        
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("基于卷积神经网络的MNIST数据分类系统 - 实验分析报告")
        report_lines.append("="*80)
        report_lines.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"训练设备: {self.device}")
        report_lines.append(f"训练轮数: {self.epochs}")
        report_lines.append(f"学习率: {self.learning_rate}")
        report_lines.append(f"批大小: {self.batch_size}")
        
        # 实验B分析
        report_lines.append("\n" + "="*80)
        report_lines.append("实验B: 网络层数对分类准确率的影响")
        report_lines.append("="*80)
        
        b_results = {k: v for k, v in self.results.items() if k.startswith('B_')}
        if b_results:
            report_lines.append("\n实验结果:")
            report_lines.append("-" * 80)
            report_lines.append(f"{'层数':<10} {'参数量':<15} {'最终准确率':<15} {'最佳准确率':<15}")
            report_lines.append("-" * 80)
            
            for result in sorted(b_results.values(), key=lambda x: x['num_conv_layers']):
                report_lines.append(
                    f"{result['num_conv_layers']:<10} "
                    f"{result['total_params']:<15,} "
                    f"{result['final_accuracy']:<15.2f} "
                    f"{result['best_accuracy']:<15.2f}"
                )
            
            # 分析结论
            best_result = max(b_results.values(), key=lambda x: x['final_accuracy'])
            report_lines.append("\n分析结论:")
            report_lines.append(f"• 最佳配置: {best_result['num_conv_layers']}层卷积")
            report_lines.append(f"• 最高准确率: {best_result['final_accuracy']:.2f}%")
            report_lines.append(f"• 参数数量: {best_result['total_params']:,}")
            
            report_lines.append("\n观察:")
            report_lines.append("• 网络层数增加时，模型的表达能力增强")
            report_lines.append("• 但层数过多可能导致过拟合或梯度消失问题")
            report_lines.append("• 需要在模型复杂度和泛化能力之间找到平衡")
        
        # 实验C分析
        report_lines.append("\n" + "="*80)
        report_lines.append("实验C: 卷积核大小对分类准确率的影响")
        report_lines.append("="*80)
        
        c_results = {k: v for k, v in self.results.items() if k.startswith('C_')}
        if c_results:
            report_lines.append("\n实验结果:")
            report_lines.append("-" * 80)
            report_lines.append(f"{'卷积核':<10} {'参数量':<15} {'最终准确率':<15} {'最佳准确率':<15}")
            report_lines.append("-" * 80)
            
            for result in sorted(c_results.values(), key=lambda x: x['kernel_size']):
                kernel_str = f"{result['kernel_size']}×{result['kernel_size']}"
                report_lines.append(
                    f"{kernel_str:<10} "
                    f"{result['total_params']:<15,} "
                    f"{result['final_accuracy']:<15.2f} "
                    f"{result['best_accuracy']:<15.2f}"
                )
            
            # 分析结论
            best_result = max(c_results.values(), key=lambda x: x['final_accuracy'])
            report_lines.append("\n分析结论:")
            report_lines.append(f"• 最佳配置: {best_result['kernel_size']}×{best_result['kernel_size']}卷积核")
            report_lines.append(f"• 最高准确率: {best_result['final_accuracy']:.2f}%")
            report_lines.append(f"• 参数数量: {best_result['total_params']:,}")
            
            report_lines.append("\n观察:")
            report_lines.append("• 较小的卷积核(3×3)能捕捉局部细节特征")
            report_lines.append("• 较大的卷积核(7×7)能捕捉更大范围的特征")
            report_lines.append("• 卷积核大小需要根据具体任务和数据特点选择")
        
        # 总体结论
        report_lines.append("\n" + "="*80)
        report_lines.append("总体结论")
        report_lines.append("="*80)
        
        all_results = list(self.results.values())
        best_overall = max(all_results, key=lambda x: x['final_accuracy'])
        
        report_lines.append(f"\n• 最佳整体配置:")
        report_lines.append(f"  - 卷积层数: {best_overall['num_conv_layers']}")
        report_lines.append(f"  - 卷积核大小: {best_overall['kernel_size']}×{best_overall['kernel_size']}")
        report_lines.append(f"  - 最终准确率: {best_overall['final_accuracy']:.2f}%")
        report_lines.append(f"  - 参数数量: {best_overall['total_params']:,}")
        
        report_lines.append("\n• 卷积神经网络在MNIST数据分类任务上表现优异")
        report_lines.append("• 通过合理设计网络结构，可以在较少参数下达到高准确率")
        report_lines.append("• 实验证明了卷积神经网络在图像分类任务中的有效性")
        
        report_lines.append("\n" + "="*80)
        
        # 保存报告
        report_text = '\n'.join(report_lines)
        print(report_text)
        
        filename = os.path.join(self.results_dir, 'analysis_report.txt')
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"\n✓ 分析报告已保存到: {filename}")
        return filename


def main():
    """主函数 - 运行完整实验"""
    print("\n" + "="*80)
    print("基于卷积神经网络的MNIST数据分类系统")
    print("="*80)
    print("\n本程序将完成以下实验:")
    print("A) 训练网络并记录每次迭代的损失值")
    print("B) 改变卷积层和池化层数，观察分类准确率")
    print("C) 改变卷积核大小，观察分类准确率")
    print("D) 分析改进的卷积神经网络数据分类的有效性")
    print("E) 设计实现基于卷积神经网络的数据分类系统")
    
    # 创建实验运行器
    runner = ExperimentRunner(epochs=10, learning_rate=0.001, batch_size=64)
    
    # 运行实验B: 网络层数对比
    runner.experiment_b_layer_comparison()
    
    # 运行实验C: 卷积核大小对比
    runner.experiment_c_kernel_comparison()
    
    # 保存结果
    runner.save_results()
    
    # 生成可视化图表
    print("\n" + "="*80)
    print("生成可视化图表...")
    print("="*80)
    runner.plot_loss_curves()
    runner.plot_accuracy_comparison()
    runner.plot_accuracy_curves()
    
    # 生成分析报告
    print("\n" + "="*80)
    print("生成分析报告...")
    print("="*80)
    runner.generate_analysis_report()
    
    print("\n" + "="*80)
    print("✓ 所有实验完成！")
    print(f"✓ 结果保存在目录: {runner.results_dir}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
