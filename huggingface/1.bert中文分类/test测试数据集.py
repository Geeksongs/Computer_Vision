from os.path import split

from datasets import load_dataset
import torch
from transformers import BertTokenizer
token=BertTokenizer.from_pretrained('bert-base-chinese')


dataset = load_dataset(
    'csv',  # 指定文件格式为 csv，即使文件后缀是 .txt，也按 CSV 格式解析
    data_files='D:\\learning\\deep_learning\\code\\huggingface\\ChnSentiCorp\\dev_modified.tsv',
    delimiter='\t',  # 如果文件是用 Tab 分隔的，改为适合的分隔符（如 ','、'\t'）
    column_names=['label', 'text'] , # 指定列名，按文件内容调整
    split='train'
)

print(dataset)

def collate_fn(data):
    print("函数被调用")
    print("当前的data为",data)
    filtered_data = []
    for sent, label in data:
        try:
            label = int(label)  # 尝试将 label 转换为整数
            filtered_data.append((sent, label))
        except ValueError:
            continue  # 如果无法转换，跳过该样本

    # 如果没有合法样本，返回空张量
    if not filtered_data:
        print("没有一个合法样本")
        return torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0)

    sents = [i[0] for i in filtered_data]
    labels = [i[1] for i in filtered_data]

    print("the length of the sents is",len(sents))
    print("the length of the labels is",len(labels))

    # 开始编码
    data = token.batch_encode_plus(
        batch_text_or_text_pairs=sents,
        add_special_tokens=True,
        padding=True,
        truncation=True,
        max_length=500,
        return_tensors='pt',
        return_length=True
    )

    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    token_type_ids = data['token_type_ids']
    labels = torch.LongTensor(labels)

    return input_ids, attention_mask, token_type_ids, labels


import torch


def collate_fn_test(data):

    # 用于存储合法样本
    filtered_data = []

    # 遍历输入数据
    for item in data:
        try:
            if item['label'] is None or item['text'] is None:
                raise ValueError("label 或 text 是 None")

            # 提取 label 和 text
            label = int(float(item['label']))  # 将 label 转换为整数
            sent = item['text']  # 提取文本内容

            # 检查文本内容是否为空
            if not sent or not isinstance(sent, str):
                raise ValueError("文本内容无效或为空")

            # 将合法样本添加到列表中
            filtered_data.append((sent, label))
        except Exception as e:
            # 捕获异常并打印详细信息
            print("当前的错误为：", e)
            print("当前的item为：", item)
            continue  # 跳过无效样本

    # 如果没有合法样本，返回空张量
    if not filtered_data:
        print("没有一个合法样本")
        return torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0)

    # 解压合法样本为文本和标签
    sents = [i[0] for i in filtered_data]  # 文本
    labels = [i[1] for i in filtered_data]  # 标签


    # 编码文本数据
    data = token.batch_encode_plus(
        batch_text_or_text_pairs=sents,
        add_special_tokens=True,
        padding=True,
        truncation=True,
        max_length=500,
        return_tensors='pt',
        return_length=True
    )

    # 转换为 PyTorch 张量
    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    token_type_ids = data['token_type_ids']
    labels = torch.LongTensor(labels)

    # 返回编码后的数据
    return input_ids, attention_mask, token_type_ids, labels


subset_dataset = dataset.select(range(300))  # 选择前 300 个样本
loader_test_we_use = torch.utils.data.DataLoader(

    dataset=subset_dataset,
    batch_size=32,
    collate_fn=collate_fn_test,
    shuffle=True,
    drop_last=True,
)

print("the length of the loader is :",len(loader_test_we_use))

print("start to print test data set")
for test_input_ids, test_attention_mask, test_token_type_ids, test_labels in loader_test_we_use:
    print(test_input_ids)






