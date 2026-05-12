import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import base64
import io
from flask import Flask, request, jsonify, render_template

# 导入UltraLightCNN - 专为MNIST优化的超轻量CNN
from src.lightweight_models import UltraLightCNN

app = Flask(__name__, template_folder='templates', static_folder='static')

# 1. 初始化和加载模型配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 使用UltraLightCNN（~33k参数，目标>98%准确率）
model = UltraLightCNN(dropout_rate=0.3).to(DEVICE)
model.eval()

# 权重路径（请确保你之前运行过训练并保存了这个文件）
MODEL_WEIGHTS_PATH = os.path.join('experiment_results', 'lightweight_cnn.pth')

# 尝试加载权重
if os.path.exists(MODEL_WEIGHTS_PATH):
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE))
    print(f"✅ 成功加载轻量化模型权重: {MODEL_WEIGHTS_PATH}")
else:
    print(f"⚠️ 警告: 未找到模型权重文件 {MODEL_WEIGHTS_PATH}。模型将使用随机初始化权重（预测结果将不准）。请先训练模型！")

def preprocess_image(base64_string):
    """将前端传来的base64图片解码、调整尺寸并转为Tensor"""
    # 移除 'data:image/png;base64,' 前缀
    image_data = base64.b64decode(base64_string.split(',')[1])
    
    # 转为 PIL Image，通常前端画板带有透明度，我们提取其灰度图并反转黑白
    # （假设前端是白底黑字，我们需要转为黑底白字以符合 MNIST）
    image = Image.open(io.BytesIO(image_data)).convert('L')
    
    # 如果前端画板是白底黑字，需要反色：image = ImageOps.invert(image)
    # 我们的前端代码将设置为黑底白字，所以不需要反色
    
    # Resize 到 28x28
    image = image.resize((28, 28))
    
    # 转换为 numpy 数组并标准化 (按照 MNIST 标准)
    img_array = np.array(image) / 255.0
    img_array = (img_array - 0.1307) / 0.3081
    
    # 转换为 PyTorch Tensor: shape [1, 1, 28, 28]
    tensor = torch.FloatTensor(img_array).unsqueeze(0).unsqueeze(0)
    return tensor

@app.route('/')
def index():
    """返回前端主页"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """处理前端发来的预测请求"""
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': '未找到图片数据'}), 400
            
        # 预处理图片
        tensor_img = preprocess_image(data['image']).to(DEVICE)
        
        # 推理
        with torch.no_grad():
            outputs = model(tensor_img)
            probabilities = F.softmax(outputs[0], dim=0)
            
            # 获取概率最高的前3个预测结果
            top3_prob, top3_idx = torch.topk(probabilities, 3)
            
            results = []
            for prob, idx in zip(top3_prob, top3_idx):
                results.append({
                    'class': str(idx.item()),
                    'probability': float(prob.item()) * 100 # 转为百分比
                })
                
        return jsonify({'success': True, 'predictions': results})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n🌐 启动 Flask Web 服务器...")
    print("👉 请在浏览器中打开: http://127.0.0.1:5000\n")
    # debug=True 会在代码修改后自动重启服务器
    app.run(host='127.0.0.1', port=5000, debug=True)