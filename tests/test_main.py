"""
单元测试示例
"""

import sys
import os

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_example():
    """示例测试"""
    assert 1 + 1 == 2
    print("测试通过!")

if __name__ == "__main__":
    test_example()
