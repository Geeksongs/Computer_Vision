from datasets import load_dataset
from datasets import Dataset
import pandas as pd
#data = pd.read_csv('D:\\learning\\deep_learning\\code\\huggingface\\ChnSentiCorp\\train.tsv', sep='\t')
#dataset = Dataset.from_pandas(data)

# 使用本地路径加载数据集
#from datasets import load_dataset

# 使用双反斜杠
#dataset = load_dataset(path='D:\\learning\\deep_learning\\code\\huggingface\\ChnSentiCorp\\train.tsv', split='train')

#print(dataset)

#import pandas as pd
#data = pd.read_csv('D:\\learning\\deep_learning\\code\\huggingface\\ChnSentiCorp\\train.tsv', sep='\t')
#dataset = Dataset.from_pandas(data)

from datasets import load_dataset

# 加载文本文件
dataset = load_dataset(
    'csv',  # 指定文件格式为 csv，即使文件后缀是 .txt，也按 CSV 格式解析
    data_files='D:\\learning\\deep_learning\\code\\huggingface\\ChnSentiCorp\\train.tsv',
    split='train',
    delimiter='\t',  # 如果文件是用 Tab 分隔的，改为适合的分隔符（如 ','、'\t'）
    column_names=['label', 'text']  # 指定列名，按文件内容调整
)

print(dataset)


# 1.排序和打乱
print(dataset['label'][:20])
for i in dataset['text'][:5]:
    print()
    print(i)


#2.选择和过滤

print(dataset.select([0,10,20,30,40,50]))

#过滤

def f(data):
    return data['text'].startswith('选择')

start_with_ar=dataset.filter(f)
print(len(start_with_ar),start_with_ar['text'])