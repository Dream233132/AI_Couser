"""
可动态配置的卷积神经网络（ConfigurableCNN）

用于处理 28×28 的 MNIST 单通道图像分类任务。
支持动态配置卷积层数量、卷积核大小等参数。
"""

import torch
import torch.nn as nn


class ConfigurableCNN(nn.Module):
    """
    可动态配置的卷积神经网络类
    
    特点：
    - 支持1、2、3层卷积
    - 支持3×3、5×5、7×7卷积核
    - 自动计算padding以保持特征图尺寸
    - 自动计算全连接层输入维度
    - 为轻量化改造预留 use_depthwise 参数
    
    参数说明：
        num_conv_layers (int): 卷积层数量，可选 1, 2, 3，默认2
        kernel_size (int): 卷积核大小，可选 3, 5, 7，默认3
        use_depthwise (bool): 是否使用深度可分离卷积（暂不生效），默认False
    
    示例：
        >>> model = ConfigurableCNN(num_conv_layers=2, kernel_size=3)
        >>> x = torch.randn(64, 1, 28, 28)
        >>> output = model(x)
        >>> print(output.shape)  # torch.Size([64, 10])
    """
    
    def __init__(self, num_conv_layers=2, kernel_size=3, use_depthwise=False):
        """
        初始化 ConfigurableCNN 网络
        
        参数说明：
            num_conv_layers (int): 卷积层数量，必须是 1, 2, 或 3
            kernel_size (int): 卷积核大小，必须是 3, 5, 或 7
            use_depthwise (bool): 预留参数，用于后续轻量化改造（当前不生效）
        
        异常：
            AssertionError: 当参数不在允许范围内时抛出
        """
        super(ConfigurableCNN, self).__init__()
        
        # ============================================
        # 第一步：参数验证
        # ============================================
        assert num_conv_layers in [1, 2, 3], \
            f"num_conv_layers must be 1, 2, or 3, got {num_conv_layers}"
        assert kernel_size in [3, 5, 7], \
            f"kernel_size must be 3, 5, or 7, got {kernel_size}"
        
        self.num_conv_layers = num_conv_layers
        self.kernel_size = kernel_size
        self.use_depthwise = use_depthwise
        
        # ============================================
        # 第二步：计算 padding（关键）
        # ============================================
        # 公式：padding = kernel_size // 2
        # 这确保卷积后特征图尺寸不变，只受池化影响
        # 验证：
        #   kernel_size=3: padding=1, output_size = (28-3+2*1)/1+1 = 28 ✓
        #   kernel_size=5: padding=2, output_size = (28-5+2*2)/1+1 = 28 ✓
        #   kernel_size=7: padding=3, output_size = (28-7+2*3)/1+1 = 28 ✓
        padding = kernel_size // 2
        
        # ============================================
        # 第三步：动态构建卷积层模块
        # ============================================
        conv_modules = []
        
        # 初始参数
        in_channels = 1              # MNIST 单通道输入
        out_channels = 32            # 第一层输出通道数
        current_feature_size = 28    # 当前特征图尺寸
        
        # 记录每层后的特征图尺寸和通道数（用于调试和计算FC层）
        self.layer_info = []
        
        for layer_idx in range(num_conv_layers):
            print(f"\n[Layer {layer_idx + 1}]")
            print(f"  Input channels: {in_channels}")
            print(f"  Output channels: {out_channels}")
            print(f"  Kernel size: {kernel_size}")
            print(f"  Padding: {padding}")
            print(f"  Input feature size: {current_feature_size}×{current_feature_size}")
            
            # 1. 卷积层
            # 保持特征图尺寸不变（通过padding）
            conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False  # 后面有BatchNorm，所以不需要bias
            )
            conv_modules.append(conv)
            
            # 2. BatchNorm2d
            # 加速收敛，增加网络稳定性
            bn = nn.BatchNorm2d(out_channels)
            conv_modules.append(bn)
            
            # 3. ReLU激活函数
            relu = nn.ReLU(inplace=True)
            conv_modules.append(relu)
            
            # 4. MaxPool2d(2×2)
            # 特征图尺寸降低为 1/2
            maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
            conv_modules.append(maxpool)
            
            # 更新特征图尺寸（池化后）
            current_feature_size = current_feature_size // 2
            print(f"  Output feature size: {current_feature_size}×{current_feature_size}")
            
            # 记录当前层信息
            self.layer_info.append({
                'layer': layer_idx + 1,
                'out_channels': out_channels,
                'feature_size': current_feature_size
            })
            
            # 为下一层做准备
            in_channels = out_channels
            # 逐层增加通道数，但不超过256
            out_channels = min(out_channels * 2, 256)
        
        # 使用 nn.Sequential 组织所有卷积模块
        self.conv_layers = nn.Sequential(*conv_modules)
        
        # ============================================
        # 第四步：动态计算全连接层输入特征数
        # ============================================
        # 最后一个卷积层的输出通道数
        last_layer_info = self.layer_info[-1]
        last_out_channels = last_layer_info['out_channels']
        final_feature_size = last_layer_info['feature_size']
        
        # Flatten 后的特征维度 = channels × height × width
        fc_input_size = last_out_channels * final_feature_size * final_feature_size
        
        print(f"\n[Fully Connected Layers]")
        print(f"  FC input size: {last_out_channels}×{final_feature_size}×{final_feature_size} = {fc_input_size}")
        
        # ============================================
        # 第五步：构建全连接层
        # ============================================
        # 中间层：减少特征数，添加非线性
        self.fc1 = nn.Linear(fc_input_size, 128)
        # 输出层：10 个类别的 logits
        self.fc2 = nn.Linear(128, 10)
        
        # ReLU 激活函数（在forward中使用）
        self.relu = nn.ReLU(inplace=True)
        
        print(f"  FC1: {fc_input_size} -> 128")
        print(f"  FC2: 128 -> 10 (logits)")
        print(f"\n✓ ConfigurableCNN 初始化完成")
        print(f"  配置: {num_conv_layers} conv layers, kernel_size={kernel_size}\n")
    
    def forward(self, x):
        """
        前向传播
        
        参数：
            x (torch.Tensor): 输入张量，形状 [batch_size, 1, 28, 28]
        
        返回：
            torch.Tensor: 输出 logits，形状 [batch_size, 10]
        
        处理流程：
            1. 通过卷积层序列提取特征
            2. Flatten 转换为一维向量
            3. 通过第一个全连接层 + ReLU
            4. 通过第二个全连接层输出 logits
        """
        # 通过所有卷积层
        # 输入: [batch_size, 1, 28, 28]
        # 输出: [batch_size, out_channels, final_feature_size, final_feature_size]
        x = self.conv_layers(x)
        
        # Flatten：转换为二维张量 [batch_size, flatten_size]
        # 使用 x.size(0) 获取 batch_size，确保兼容不同的 batch 大小
        x = x.view(x.size(0), -1)
        
        # 第一个全连接层 + ReLU
        x = self.fc1(x)
        x = self.relu(x)
        
        # 第二个全连接层输出 logits
        # 不在这里应用 softmax，通常在 loss 函数中处理
        x = self.fc2(x)
        
        return x
    
    def get_config_summary(self):
        """
        获取网络配置的文字总结
        
        返回：
            str: 网络配置描述
        """
        summary = f"""
ConfigurableCNN Configuration
============================
Num Conv Layers: {self.num_conv_layers}
Kernel Size: {self.kernel_size}×{self.kernel_size}
Padding: {self.kernel_size // 2}
Use Depthwise: {self.use_depthwise}

Layer Information:
"""
        for info in self.layer_info:
            summary += f"  Layer {info['layer']}: channels={info['out_channels']}, feature_size={info['feature_size']}×{info['feature_size']}\n"
        
        return summary


def count_parameters(model):
    """
    计算模型的可训练参数总数
    
    参数：
        model (nn.Module): PyTorch 模型
    
    返回：
        int: 可训练参数总数
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_model_output_shapes():
    """
    测试模型输出形状的正确性
    
    验证不同配置下的模型输出形状是否正确
    """
    print("\n" + "="*70)
    print("测试模型输出形状")
    print("="*70)
    
    # 测试所有有效配置组合
    configs = [
        (1, 3), (1, 5), (1, 7),
        (2, 3), (2, 5), (2, 7),
        (3, 3), (3, 5), (3, 7),
    ]
    
    for num_layers, kernel_sz in configs:
        print(f"\n✓ 配置: num_conv_layers={num_layers}, kernel_size={kernel_sz}")
        
        # 创建模型
        model = ConfigurableCNN(num_conv_layers=num_layers, kernel_size=kernel_sz)
        
        # 创建随机输入 [batch_size=4, channels=1, height=28, width=28]
        x = torch.randn(4, 1, 28, 28)
        
        # 前向传播
        output = model(x)
        
        # 验证输出形状
        assert output.shape == torch.Size([4, 10]), \
            f"Expected output shape [4, 10], got {output.shape}"
        
        # 计算参数数量
        total_params = count_parameters(model)
        print(f"  输出形状: {output.shape}")
        print(f"  参数总数: {total_params:,}")


if __name__ == "__main__":
    # 演示模型使用
    print("\n" + "="*70)
    print("ConfigurableCNN 模型演示")
    print("="*70)
    
    # 创建模型
    print("\n[示例1] 创建 2 层卷积网络，卷积核大小为 3×3")
    model1 = ConfigurableCNN(num_conv_layers=2, kernel_size=3)
    print(model1.get_config_summary())
    
    # 创建输入
    x = torch.randn(4, 1, 28, 28)
    print(f"输入形状: {x.shape}")
    
    # 前向传播
    output = model1(x)
    print(f"输出形状: {output.shape}")
    print(f"参数总数: {count_parameters(model1):,}")
    
    # 其他配置示例
    print("\n[示例2] 创建 1 层卷积网络，卷积核大小为 7×7")
    model2 = ConfigurableCNN(num_conv_layers=1, kernel_size=7)
    output2 = model2(x)
    print(f"输出形状: {output2.shape}")
    print(f"参数总数: {count_parameters(model2):,}")
    
    print("\n[示例3] 创建 3 层卷积网络，卷积核大小为 5×5")
    model3 = ConfigurableCNN(num_conv_layers=3, kernel_size=5)
    output3 = model3(x)
    print(f"输出形状: {output3.shape}")
    print(f"参数总数: {count_parameters(model3):,}")
    
    # 测试所有配置
    test_model_output_shapes()
    
    print("\n" + "="*70)
    print("✓ 所有测试通过！")
    print("="*70 + "\n")
