"""
可解释性分析模块

实现Grad-CAM (Gradient-weighted Class Activation Mapping)
用于生成模型注意力热力图，分析模型关注的图像区域
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict
import cv2


class GradCAM:
    """
    Grad-CAM: 基于梯度的类激活映射
    
    原理：
    1. 获取目标卷积层的特征图和梯度
    2. 计算梯度的全局平均池化作为权重
    3. 加权求和特征图得到热力图
    4. ReLU激活并归一化
    
    参数：
        model: 目标模型
        target_layer: 目标卷积层（用于提取特征图）
    """
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # 注册钩子函数
        self._register_hooks()
    
    def _register_hooks(self):
        """注册前向和反向传播钩子"""
        
        def forward_hook(module, input, output):
            """前向传播钩子：保存激活值"""
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            """反向传播钩子：保存梯度"""
            self.gradients = grad_output[0].detach()
        
        # 注册钩子
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate_cam(self, input_image: torch.Tensor, target_class: int = None) -> np.ndarray:
        """
        生成类激活映射（CAM）
        
        参数：
            input_image: 输入图像 [1, C, H, W]
            target_class: 目标类别（None表示使用预测类别）
        
        返回：
            cam: 热力图 [H, W]，值范围[0, 1]
        """
        self.model.eval()
        
        # 前向传播
        output = self.model(input_image)
        
        # 确定目标类别
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # 反向传播
        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward()
        
        # 获取梯度和激活值
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        
        # 计算权重：对梯度进行全局平均池化
        weights = gradients.mean(dim=(1, 2))  # [C]
        
        # 加权求和
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)  # [H, W]
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # ReLU激活（只保留正值）
        cam = F.relu(cam)
        
        # 归一化到[0, 1]
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam
    
    def generate_heatmap(self, input_image: torch.Tensor, target_class: int = None,
                        original_size: Tuple[int, int] = (28, 28)) -> np.ndarray:
        """
        生成热力图并调整到原始图像尺寸
        
        参数：
            input_image: 输入图像
            target_class: 目标类别
            original_size: 原始图像尺寸 (H, W)
        
        返回：
            heatmap: 彩色热力图 [H, W, 3]，RGB格式
        """
        # 生成CAM
        cam = self.generate_cam(input_image, target_class)
        
        # 调整到原始尺寸
        cam_resized = cv2.resize(cam, original_size)
        
        # 转换为彩色热力图（使用JET colormap）
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        return heatmap
    
    def overlay_heatmap(self, input_image: torch.Tensor, target_class: int = None,
                       alpha: float = 0.4) -> np.ndarray:
        """
        将热力图叠加到原始图像上
        
        参数：
            input_image: 输入图像 [1, 1, H, W]
            target_class: 目标类别
            alpha: 热力图透明度
        
        返回：
            overlayed: 叠加后的图像 [H, W, 3]
        """
        # 生成热力图
        heatmap = self.generate_heatmap(input_image, target_class)
        
        # 将输入图像转换为RGB格式
        img = input_image[0, 0].cpu().numpy()
        
        # 反归一化（假设使用MNIST标准化）
        img = img * 0.3081 + 0.1307
        img = np.clip(img, 0, 1)
        
        # 转换为RGB
        img_rgb = np.stack([img, img, img], axis=-1)
        img_rgb = (img_rgb * 255).astype(np.uint8)
        
        # 叠加
        overlayed = cv2.addWeighted(img_rgb, 1 - alpha, heatmap, alpha, 0)
        
        return overlayed


def analyze_layer_attention(model: nn.Module, test_loader, device: torch.device,
                           num_samples: int = 10) -> Dict:
    """
    分析不同卷积层的注意力模式
    
    参数：
        model: 目标模型
        test_loader: 测试数据加载器
        device: 计算设备
        num_samples: 分析样本数量
    
    返回：
        results: 包含每层热力图的字典
    """
    model = model.to(device)
    model.eval()
    
    # 找到所有卷积层
    conv_layers = []
    layer_names = []
    
    def find_conv_layers(module, prefix=''):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Conv2d):
                conv_layers.append(child)
                layer_names.append(full_name)
            else:
                find_conv_layers(child, full_name)
    
    find_conv_layers(model)
    
    print(f"\n找到 {len(conv_layers)} 个卷积层:")
    for i, name in enumerate(layer_names):
        print(f"  Layer {i+1}: {name}")
    
    # 收集样本
    images_list = []
    labels_list = []
    
    for images, labels in test_loader:
        for i in range(min(images.size(0), num_samples - len(images_list))):
            images_list.append(images[i:i+1])
            labels_list.append(labels[i].item())
        
        if len(images_list) >= num_samples:
            break
    
    # 为每个卷积层生成热力图
    results = {
        'layer_names': layer_names,
        'heatmaps': {},  # {layer_idx: [heatmaps for all samples]}
        'images': images_list,
        'labels': labels_list
    }
    
    for layer_idx, (layer, layer_name) in enumerate(zip(conv_layers, layer_names)):
        print(f"\n分析 {layer_name}...")
        
        gradcam = GradCAM(model, layer)
        layer_heatmaps = []
        
        for img in images_list:
            img = img.to(device)
            heatmap = gradcam.generate_heatmap(img)
            layer_heatmaps.append(heatmap)
        
        results['heatmaps'][layer_idx] = layer_heatmaps
    
    return results


def compare_architecture_attention(models_dict: Dict[str, nn.Module], 
                                   test_loader, device: torch.device,
                                   num_samples: int = 5) -> Dict:
    """
    对比不同网络结构的注意力模式
    
    参数：
        models_dict: 模型字典 {model_name: model}
        test_loader: 测试数据加载器
        device: 计算设备
        num_samples: 对比样本数量
    
    返回：
        comparison_results: 对比结果字典
    """
    # 收集测试样本
    images_list = []
    labels_list = []
    
    for images, labels in test_loader:
        for i in range(min(images.size(0), num_samples - len(images_list))):
            images_list.append(images[i:i+1])
            labels_list.append(labels[i].item())
        
        if len(images_list) >= num_samples:
            break
    
    # 为每个模型生成热力图
    comparison_results = {
        'images': images_list,
        'labels': labels_list,
        'models': {}
    }
    
    for model_name, model in models_dict.items():
        print(f"\n分析模型: {model_name}")
        model = model.to(device)
        model.eval()
        
        # 找到最后一个卷积层
        last_conv = None
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        
        if last_conv is None:
            print(f"  警告: {model_name} 没有卷积层")
            continue
        
        # 生成热力图
        gradcam = GradCAM(model, last_conv)
        heatmaps = []
        predictions = []
        
        for img in images_list:
            img = img.to(device)
            
            # 预测
            with torch.no_grad():
                output = model(img)
                pred = output.argmax(dim=1).item()
                predictions.append(pred)
            
            # 生成热力图
            heatmap = gradcam.generate_heatmap(img)
            heatmaps.append(heatmap)
        
        comparison_results['models'][model_name] = {
            'heatmaps': heatmaps,
            'predictions': predictions
        }
    
    return comparison_results


def calculate_attention_metrics(heatmap: np.ndarray, threshold: float = 0.5) -> Dict:
    """
    计算注意力热力图的统计指标
    
    参数：
        heatmap: 热力图 [H, W]，值范围[0, 1]
        threshold: 高注意力区域阈值
    
    返回：
        metrics: 统计指标字典
    """
    # 转换为灰度（如果是彩色）
    if len(heatmap.shape) == 3:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2GRAY) / 255.0
    
    # 计算指标
    metrics = {
        'mean_attention': float(heatmap.mean()),
        'max_attention': float(heatmap.max()),
        'std_attention': float(heatmap.std()),
        'high_attention_ratio': float((heatmap > threshold).sum() / heatmap.size),
        'center_attention': float(heatmap[10:18, 10:18].mean()),  # 中心区域
        'edge_attention': float(np.concatenate([
            heatmap[0, :], heatmap[-1, :], 
            heatmap[:, 0], heatmap[:, -1]
        ]).mean())  # 边缘区域
    }
    
    return metrics
