# Python 项目 Copilot 指引

本项目是一个基础的Python项目框架，用于开发各种应用。

## 项目概述

- **项目类型**: Python 项目
- **主入口**: `src/main.py`
- **虚拟环境**: `venv/`

## 快速开始

1. 创建虚拟环境: `python -m venv venv`
2. 激活虚拟环境: `venv\Scripts\activate` (Windows) 或 `source venv/bin/activate` (Linux/Mac)
3. 安装依赖: `pip install -r requirements.txt`
4. 运行项目: `python src/main.py`

## 代码规范

- 遵循 PEP 8 风格指南
- 使用有意义的变量和函数名称
- 添加适当的注释和文档字符串
- 编写单元测试

## 常用命令

```bash
# 运行主程序
python src/main.py

# 运行测试
python -m pytest tests/

# 安装新依赖
pip install <package-name>

# 更新依赖列表
pip freeze > requirements.txt
```
