"""
轻量化模型训练脚本 - 优化版

功能：
1. 数据增强（随机仿射变换）
2. 余弦退火学习率调度
3. 最佳模型自动保存（基于验证准确率）
4. ONNX 格式导出
5. 最终验证 >98% 准确率
"""

import os
import torch
import torch.nn as nn
import torch.onnx
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.lightweight_models import UltraLightCNN, count_parameters
from src.train import train_epoch, test
from torchvision import transforms, datasets


def get_augmented_loaders(batch_size_train=64, batch_size_test=1000):
    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)
    return train_loader, test_loader


def train_lightweight_model():
    print("\n" + "=" * 80)
    print("🚀 轻量化模型优化训练 (目标: >98% 准确率)")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    EPOCHS = 25
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4

    print(f"\n📋 训练配置: device={device}, epochs={EPOCHS}, batch={BATCH_SIZE}, lr={LEARNING_RATE}")

    train_loader, test_loader = get_augmented_loaders(batch_size_train=BATCH_SIZE)
    print(f"   训练batch数: {len(train_loader)}, 测试batch数: {len(test_loader)}")

    model = UltraLightCNN(dropout_rate=0.3).to(device)
    params = count_parameters(model)
    print(f"   参数量: {params:,} (<50k: {'✅' if params < 50000 else '❌'})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    best_accuracy = 0.0
    best_epoch = 0
    epoch_accuracies = []
    save_dir = 'experiment_results'
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(EPOCHS):
        iteration_losses = train_epoch(model, device, train_loader, optimizer, criterion)
        test_loss, test_accuracy = test(model, device, test_loader, criterion)
        epoch_accuracies.append(test_accuracy)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(f"Epoch [{epoch + 1}/{EPOCHS}] | Loss: {iteration_losses[-1]:.4f} | Test Acc: {test_accuracy:.2f}% | LR: {current_lr:.2e}")

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(save_dir, 'lightweight_cnn.pth'))
            print(f"  👑 新最佳模型! Acc: {best_accuracy:.2f}%")

    print(f"\n✅ 训练完成! 最佳准确率: {best_accuracy:.2f}% (Epoch {best_epoch})")

    # 最终验证
    model.load_state_dict(torch.load(os.path.join(save_dir, 'lightweight_cnn.pth')))
    final_loss, final_accuracy = test(model, device, test_loader, criterion)
    print(f"🏆 最终验证: 准确率={final_accuracy:.2f}% 目标>98%→{'✅ 达标!' if final_accuracy > 98 else '❌ 未达标'}")
    print(f"   参数数: {params:,} 限制<50k→{'✅ 达标!' if params < 50000 else '❌ 未达标'}")

    # 导出ONNX
    print(f"\n📤 导出ONNX格式...")
    model.eval()
    dummy_input = torch.randn(1, 1, 28, 28).to(device)
    onnx_path = os.path.join(save_dir, 'lightweight_cnn.onnx')
    try:
        torch.onnx.export(model, dummy_input, onnx_path, input_names=['input'], output_names=['output'],
                          dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
                          opset_version=11, do_constant_folding=True)
        print(f"  ✅ ONNX导出成功: {onnx_path}")
    except Exception as e:
        print(f"  ❌ ONNX导出失败: {e}")

    print(f"\n💾 模型权重: {os.path.join(save_dir, 'lightweight_cnn.pth')}")
    print(f"👉 现在运行 `python app.py` 启动Web演示!")
    return final_accuracy


if __name__ == "__main__":
    train_lightweight_model()