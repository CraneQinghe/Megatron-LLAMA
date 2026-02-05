import json

# 指定你的 JSON 文件路径
json_file_path = '/data/haiqwa/zevin_nfs/dataset/bookcorpus/train_data.json'


# 逐行读取 JSON 文件并输出键值
with open(json_file_path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 100:  # 只打印前 5 个条目
            break
        try:
            data = json.loads(line)
            print(f"Entry {i + 1} keys:")
            print(list(data.keys()))
            print("\n")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON on line {i + 1}: {e}")


