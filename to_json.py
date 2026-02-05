from datasets import load_from_disk
import datasets
# train_data = load_from_disk('/data/haiqwa/zevin_nfs/dataset/bookcorpus/')
# 如果需要访问 train 数据，可以直接使用
#print(train_data[0])  # 查看第一个示例
train_data = datasets.load_dataset('/data/haiqwa/zevin_nfs/dataset/bookcorpus/', split='train')
train_data.to_json("/data/haiqwa/zevin_nfs/dataset/bookcorpus/train_data.json", lines=True)