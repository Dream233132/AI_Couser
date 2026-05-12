"""
轻量化模型 - 创新要求2

设计参数量<50k的CNN模型，准确率>98%
适合移动端部署
包含多个轻量化架构：
1. UltraLightCNN - 专为MNIST优化的超轻量CNN（推荐，~33k参数）
2. MobileNetV2MNIST - 适配MNIST的MobileNetV2
3. TinyNetMNIST - 极致轻量化网络
"""

import torch
import torch.nn as nn


# ==========================================================================
# UltraLightCNN - 专为MNIST深度优化的超轻量CNN（⭐ 推荐使用！）
# ==========================================================================

class ConvBlock(nn.Module):
    """
    标准卷积块：Conv2d + BatchNorm + ReLU
    
    参数量：C_in × C_out × 3 × 3 (Conv) + C_out × 2 (BN)
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class UltraLightCNN(nn.Module):
    """
    为MNIST深度优化的超轻量卷积神经网络 ⭐
    
    ────────────────────────────────────────────
    参数量计算（所有Conv使用bias=False + BN）
    ────────────────────────────────────────────
    Block1 (28×28):
      Conv(1→12) + BN + Conv(12→12) + BN + MaxPool
      = (108+24) + (1296+24) = 1,452
    
    Block2 (14×14):
      Conv(12→24) + BN + Conv(24→24) + BN + MaxPool
      = (2592+48) + (5184+48) = 7,872
    
    Block3 (7×7):
      Conv(24→40) + BN + Conv(40→40) + BN
      = (8640+80) + (14400+80) = 23,200
    
    Classifier:
      Dropout + Linear(40→10) = 400+10 = 410
    
    总计: 1,452 + 7,872 + 23,200 + 410 = 32,934 ✅ (< 50k)
    ────────────────────────────────────────────
    
    用法：
        >>> model = UltraLightCNN()
        >>> x = torch.randn(64, 1, 28, 28)
        >>> output = model(x)
    """
    
    def __init__(self, num_classes=10, dropout_rate=0.3):
        super(UltraLightCNN, self).__init__()
        
        # Block 1: 28×28 → 14×14（基础特征提取）
        self.block1 = nn.Sequential(
            ConvBlock(1, 12, 3, 1, 1),
            ConvBlock(12, 12, 3, 1, 1),
            nn.MaxPool2d(2, 2),
        )
        
        # Block 2: 14×14 → 7×7（中等层次特征）
        self.block2 = nn.Sequential(
            ConvBlock(12, 24, 3, 1, 1),
            ConvBlock(24, 24, 3, 1, 1),
            nn.MaxPool2d(2, 2),
        )
        
        # Block 3: 7×7（高层次特征）
        self.block3 = nn.Sequential(
            ConvBlock(24, 40, 3, 1, 1),
            ConvBlock(40, 40, 3, 1, 1),
        )
        
        # 分类头
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(40, num_classes)
        
        self._initialize_weights()
        
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"✓ UltraLightCNN 初始化完成 ⭐")
        print(f"  参数量: {total_params:,} (目标: <50,000 ✅)")
        print(f"  架构: 6层Conv → 2次Pool → GAP → FC")
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


# ==========================================================================
# 工具函数
# ==========================================================================

def count_parameters(model):
    """计算模型的可训练参数总数"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_lightweight_models():
    """测试所有轻量化模型"""
    print("\n" + "="*70)
    print("轻量化模型测试")
    print("="*70)
    
    models = {
        'UltraLightCNN ⭐': UltraLightCNN(),
    }
    
    x = torch.randn(4, 1, 28, 28)
    
    for name, model in models.items():
        print(f"\n{name}:")
        output = model(x)
        params = count_parameters(model)
        print(f"  输出形状: {output.shape}")
        print(f"  参数总数: {params:,}")
        print(f"  小于50k: {'✅' if params < 50000 else '❌'}")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    test_lightweight_models()