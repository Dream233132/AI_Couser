"""
高级卷积神经网络模型

实现现代改进结构：
1. SENet-CNN: 带有Squeeze-and-Excitation注意力机制的CNN
2. DilatedCNN: 使用空洞卷积的CNN
3. DepthwiseCNN: 使用深度可分离卷积的CNN
"""

import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block (SE模块)
    
    通过全局平均池化和两个全连接层学习通道注意力权重，
    自适应地重新校准通道特征响应。
    
    参数：
        channels (int): 输入通道数
        reduction (int): 降维比例，默认16
    """
    
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        # Squeeze: 全局平均池化 [B, C, H, W] -> [B, C, 1, 1]
        y = self.squeeze(x).view(batch_size, channels)
        # Excitation: 学习通道权重 [B, C] -> [B, C]
        y = self.excitation(y).view(batch_size, channels, 1, 1)
        # Scale: 重新校准特征图
        return x * y.expand_as(x)


class SENetCNN(nn.Module):
    """
    带有SE注意力机制的卷积神经网络
    
    在每个卷积块后添加SE模块，增强通道特征表达能力。
    参数量略有增加，但准确率通常有提升。
    """
    
    def __init__(self, num_conv_layers=2, reduction=16):
        super(SENetCNN, self).__init__()
        
        self.num_conv_layers = num_conv_layers
        
        # 构建卷积层
        conv_modules = []
        in_channels = 1
        out_channels = 32
        current_size = 28
        
        for i in range(num_conv_layers):
            # 卷积层
            conv_modules.append(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False))
            conv_modules.append(nn.BatchNorm2d(out_channels))
            conv_modules.append(nn.ReLU(inplace=True))
            
            # SE模块
            conv_modules.append(SEBlock(out_channels, reduction))
            
            # 池化层
            conv_modules.append(nn.MaxPool2d(2, 2))
            
            current_size = current_size // 2
            in_channels = out_channels
            out_channels = min(out_channels * 2, 256)
        
        self.conv_layers = nn.Sequential(*conv_modules)
        
        # 全连接层
        fc_input_size = in_channels * current_size * current_size
        self.fc1 = nn.Linear(fc_input_size, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU(inplace=True)
        
        print(f"✓ SENetCNN初始化完成: {num_conv_layers}层卷积 + SE模块")
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DilatedCNN(nn.Module):
    """
    使用空洞卷积的CNN
    
    空洞卷积通过在卷积核元素之间插入空洞来扩大感受野，
    不增加参数量和计算量的情况下捕捉更大范围的上下文信息。
    """
    
    def __init__(self, num_conv_layers=2):
        super(DilatedCNN, self).__init__()
        
        self.num_conv_layers = num_conv_layers
        
        # 构建卷积层
        conv_modules = []
        in_channels = 1
        out_channels = 32
        current_size = 28
        
        # 空洞率递增
        dilation_rates = [1, 2, 3]
        
        for i in range(num_conv_layers):
            dilation = dilation_rates[i] if i < len(dilation_rates) else 1
            # 计算padding以保持尺寸
            padding = dilation
            
            # 空洞卷积层
            conv_modules.append(nn.Conv2d(
                in_channels, out_channels, 
                kernel_size=3, 
                padding=padding, 
                dilation=dilation,
                bias=False
            ))
            conv_modules.append(nn.BatchNorm2d(out_channels))
            conv_modules.append(nn.ReLU(inplace=True))
            
            # 池化层
            conv_modules.append(nn.MaxPool2d(2, 2))
            
            current_size = current_size // 2
            in_channels = out_channels
            out_channels = min(out_channels * 2, 256)
        
        self.conv_layers = nn.Sequential(*conv_modules)
        
        # 全连接层
        fc_input_size = in_channels * current_size * current_size
        self.fc1 = nn.Linear(fc_input_size, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU(inplace=True)
        
        print(f"✓ DilatedCNN初始化完成: {num_conv_layers}层空洞卷积")
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DepthwiseSeparableConv(nn.Module):
    """
    深度可分离卷积模块
    
    将标准卷积分解为：
    1. Depthwise卷积：每个输入通道独立卷积
    2. Pointwise卷积：1×1卷积进行通道融合
    
    参数量和计算量大幅减少（约8-9倍）
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        
        # Depthwise卷积：每个通道独立卷积
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, 
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,  # 关键：groups=in_channels
            bias=False
        )
        
        # Pointwise卷积：1×1卷积融合通道
        self.pointwise = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1,
            bias=False
        )
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DepthwiseCNN(nn.Module):
    """
    使用深度可分离卷积的轻量级CNN
    
    大幅减少参数量和计算量，适合移动端部署。
    在MNIST上仍能保持较高准确率。
    """
    
    def __init__(self, num_conv_layers=2):
        super(DepthwiseCNN, self).__init__()
        
        self.num_conv_layers = num_conv_layers
        
        # 构建卷积层
        conv_modules = []
        in_channels = 1
        out_channels = 32
        current_size = 28
        
        # 第一层使用标准卷积（输入通道为1）
        conv_modules.append(nn.Conv2d(1, 32, 3, padding=1, bias=False))
        conv_modules.append(nn.BatchNorm2d(32))
        conv_modules.append(nn.ReLU(inplace=True))
        conv_modules.append(nn.MaxPool2d(2, 2))
        
        current_size = current_size // 2
        in_channels = 32
        
        # 后续层使用深度可分离卷积
        for i in range(1, num_conv_layers):
            out_channels = min(in_channels * 2, 256)
            
            conv_modules.append(DepthwiseSeparableConv(
                in_channels, out_channels,
                kernel_size=3, padding=1
            ))
            conv_modules.append(nn.MaxPool2d(2, 2))
            
            current_size = current_size // 2
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*conv_modules)
        
        # 全连接层
        fc_input_size = in_channels * current_size * current_size
        self.fc1 = nn.Linear(fc_input_size, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU(inplace=True)
        
        print(f"✓ DepthwiseCNN初始化完成: {num_conv_layers}层（含深度可分离卷积）")
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def count_parameters(model):
    """计算模型的可训练参数总数"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 测试所有模型
    print("\n" + "="*70)
    print("高级模型测试")
    print("="*70)
    
    models = {
        'SENet-CNN': SENetCNN(num_conv_layers=2),
        'DilatedCNN': DilatedCNN(num_conv_layers=2),
        'DepthwiseCNN': DepthwiseCNN(num_conv_layers=2)
    }
    
    x = torch.randn(4, 1, 28, 28)
    
    for name, model in models.items():
        print(f"\n{name}:")
        output = model(x)
        params = count_parameters(model)
        print(f"  输出形状: {output.shape}")
        print(f"  参数总数: {params:,}")
    
    print("\n" + "="*70)
    print("✓ 所有模型测试通过！")
    print("="*70 + "\n")
