"""
快速测试脚本 - 验证环境配置和基本功能

本脚本用于快速验证：
1. Python环境和依赖包是否正确安装
2. GPU/CUDA是否可用
3. MNIST数据集是否可以正常加载
4. 模型是否可以正常创建和训练
5. 基本的训练和测试流程是否正常

预计运行时间: 2-3分钟
"""

import sys
import torch
import torchvision
import numpy as np
import matplotlib
import time
from datetime import datetime

def print_section(title):
    """打印分节标题"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

def check_python_version():
    """检查Python版本"""
    print_section("1. Python版本检查")
    version = sys.version_info
    print(f"Python版本: {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 7:
        print("✅ Python版本符合要求 (>= 3.7)")
        return True
    else:
        print("❌ Python版本过低，需要 >= 3.7")
        return False

def check_packages():
    """检查依赖包"""
    print_section("2. 依赖包检查")
    
    packages = {
        'torch': torch.__version__,
        'torchvision': torchvision.__version__,
        'numpy': np.__version__,
        'matplotlib': matplotlib.__version__,
    }
    
    all_ok = True
    for name, version in packages.items():
        print(f"✅ {name:15s} {version}")
    
    return all_ok

def check_gpu():
    """检查GPU配置"""
    print_section("3. GPU/CUDA检查")
    
    if torch.cuda.is_available():
        print(f"✅ CUDA可用")
        print(f"   GPU设备: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA版本: {torch.version.cuda}")
        print(f"   GPU数量: {torch.cuda.device_count()}")
        
        # 显示显存信息
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"   总显存: {total_memory:.2f} GB")
        
        # 检查cuDNN
        if torch.backends.cudnn.is_available():
            print(f"✅ cuDNN可用 (版本: {torch.backends.cudnn.version()})")
        else:
            print(f"⚠️  cuDNN不可用")
        
        return 'cuda'
    else:
        print("⚠️  CUDA不可用，将使用CPU训练")
        print("   提示: 安装CUDA版本的PyTorch可以大幅提升训练速度")
        return 'cpu'

def test_data_loading():
    """测试数据加载"""
    print_section("4. 数据加载测试")
    
    try:
        from src.main import get_data_loaders
        
        print("正在加载MNIST数据集...")
        start_time = time.time()
        train_loader, test_loader = get_data_loaders(batch_size_train=64, batch_size_test=1000)
        load_time = time.time() - start_time
        
        print(f"✅ 数据加载成功 (耗时: {load_time:.2f}秒)")
        print(f"   训练集批次数: {len(train_loader)}")
        print(f"   测试集批次数: {len(test_loader)}")
        
        # 获取一个batch测试
        data, target = next(iter(train_loader))
        print(f"   单个batch形状: {data.shape}")
        print(f"   标签形状: {target.shape}")
        
        return train_loader, test_loader
    except Exception as e:
        print(f"❌ 数据加载失败: {str(e)}")
        return None, None

def test_model_creation():
    """测试模型创建"""
    print_section("5. 模型创建测试")
    
    try:
        from src.models import ConfigurableCNN, count_parameters
        
        # 创建一个简单的2层CNN
        print("创建2层卷积网络...")
        model = ConfigurableCNN(num_conv_layers=2, kernel_size=3)
        
        # 计算参数量
        total_params = count_parameters(model)
        print(f"✅ 模型创建成功")
        print(f"   模型参数量: {total_params:,}")
        
        # 测试前向传播
        dummy_input = torch.randn(1, 1, 28, 28)
        output = model(dummy_input)
        print(f"   输入形状: {dummy_input.shape}")
        print(f"   输出形状: {output.shape}")
        
        return model
    except Exception as e:
        print(f"❌ 模型创建失败: {str(e)}")
        return None

def test_training(model, train_loader, test_loader, device):
    """测试训练流程"""
    print_section("6. 训练流程测试")
    
    if model is None or train_loader is None:
        print("❌ 跳过训练测试（前置条件未满足）")
        return False
    
    try:
        from src.train import train_epoch, test
        import torch.nn as nn
        import torch.optim as optim
        
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        print(f"开始快速训练测试（1个epoch，仅前100个batch）...")
        print(f"训练设备: {device}")
        
        # 训练1个epoch（限制batch数量）
        start_time = time.time()
        model.train()
        total_loss = 0
        batch_count = 0
        max_batches = 100  # 只训练100个batch
        
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx >= max_batches:
                break
                
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
            # 每20个batch显示一次进度
            if (batch_idx + 1) % 20 == 0:
                avg_loss = total_loss / batch_count
                print(f"   Batch [{batch_idx+1}/{max_batches}], 平均损失: {avg_loss:.4f}")
        
        train_time = time.time() - start_time
        avg_loss = total_loss / batch_count
        
        print(f"✅ 训练测试完成")
        print(f"   训练时间: {train_time:.2f}秒")
        print(f"   平均损失: {avg_loss:.4f}")
        print(f"   训练速度: {batch_count/train_time:.2f} batch/秒")
        
        # 测试评估
        print("\n开始测试评估...")
        start_time = time.time()
        accuracy = test(model, test_loader, device)
        test_time = time.time() - start_time
        
        print(f"✅ 评估测试完成")
        print(f"   测试准确率: {accuracy:.2f}%")
        print(f"   测试时间: {test_time:.2f}秒")
        
        # 判断是否正常
        if accuracy > 50:  # 快速训练后准确率应该 > 50%
            print(f"✅ 模型训练正常（准确率 > 50%）")
            return True
        else:
            print(f"⚠️  准确率较低，可能需要更多训练")
            return True  # 仍然算通过，因为流程是正常的
            
    except Exception as e:
        print(f"❌ 训练测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_visualization():
    """测试可视化功能"""
    print_section("7. 可视化功能测试")
    
    try:
        import matplotlib.pyplot as plt
        
        # 测试创建简单图表
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.plot([1, 2, 3, 4], [1, 4, 2, 3])
        ax.set_title('测试图表')
        plt.close(fig)
        
        print("✅ Matplotlib可视化功能正常")
        return True
    except Exception as e:
        print(f"⚠️  可视化测试失败: {str(e)}")
        return False

def print_summary(results):
    """打印测试总结"""
    print_section("测试总结")
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"\n测试项目: {total}")
    print(f"通过: {passed}")
    print(f"失败: {total - passed}")
    
    print("\n详细结果:")
    for name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {status}  {name}")
    
    if passed == total:
        print("\n" + "="*80)
        print("🎉 所有测试通过！环境配置正确，可以开始实验。")
        print("="*80)
        print("\n下一步:")
        print("  1. 运行基础实验: python experiments.py")
        print("  2. 运行高级实验: python advanced_experiments.py")
        print("  3. 训练轻量化模型: python train_lightweight.py")
        print("  4. 启动Web演示: python app.py")
    else:
        print("\n" + "="*80)
        print("⚠️  部分测试未通过，请检查环境配置。")
        print("="*80)
        print("\n建议:")
        print("  1. 检查Python版本 (>= 3.7)")
        print("  2. 重新安装依赖: pip install -r requirements.txt")
        print("  3. 如果使用GPU，确保安装了CUDA版本的PyTorch")

def main():
    """主函数"""
    print("\n" + "="*80)
    print("  基于卷积神经网络的MNIST数据分类系统 - 快速测试")
    print("="*80)
    print(f"\n开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("预计耗时: 2-3分钟")
    
    results = {}
    
    # 1. Python版本检查
    results['Python版本'] = check_python_version()
    
    # 2. 依赖包检查
    results['依赖包'] = check_packages()
    
    # 3. GPU检查
    device = check_gpu()
    results['GPU/CUDA'] = True  # 无论是否有GPU都算通过
    
    # 4. 数据加载测试
    train_loader, test_loader = test_data_loading()
    results['数据加载'] = train_loader is not None
    
    # 5. 模型创建测试
    model = test_model_creation()
    results['模型创建'] = model is not None
    
    # 6. 训练流程测试
    if model is not None and train_loader is not None:
        results['训练流程'] = test_training(model, train_loader, test_loader, device)
    else:
        results['训练流程'] = False
    
    # 7. 可视化测试
    results['可视化功能'] = test_visualization()
    
    # 打印总结
    print_summary(results)
    
    print(f"\n结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断测试")
    except Exception as e:
        print(f"\n\n❌ 测试过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
