import torch
import types

def inspect_pth(path):
    print(f"🔍 正在检查文件: {path}")
    try:
        obj = torch.load(path, map_location='cpu')
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return

    print(f"📦 顶层类型: {type(obj)}")

    if isinstance(obj, torch.Tensor):
        print(f"✅ 是 Tensor，形状: {obj.shape}")

    elif isinstance(obj, torch.nn.Embedding):
        print(f"✅ 是 nn.Embedding，权重形状: {obj.weight.shape}")

    elif isinstance(obj, dict):
        print("📚 是 dict，键值对如下:")
        for k, v in obj.items():
            print(f"  - 键: {k}, 类型: {type(v)}", end='')
            if isinstance(v, torch.Tensor):
                print(f"，形状: {v.shape}")
            elif isinstance(v, (list, tuple)):
                print(f"，长度: {len(v)}")
            elif isinstance(v, dict):
                print(f"，字典大小: {len(v)}")
            else:
                print()

    elif isinstance(obj, (list, tuple)):
        print(f"📋 是 {type(obj)}，长度: {len(obj)}")
        for i, item in enumerate(obj):
            print(f"  - 第 {i} 项: 类型: {type(item)}", end='')
            if isinstance(item, torch.Tensor):
                print(f"，形状: {item.shape}")
            else:
                print()

    else:
        print("⚠️ 无法识别的类型，建议手动调试 obj 结构")

# 调用函数
inspect_pth('/workspace/ACE /save_model/tweet_groups.pth')  # 替换为你的 .pth 文件路径
