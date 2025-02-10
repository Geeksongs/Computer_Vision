#1. step1：定义数据集

import torch
from datasets import load_dataset

#2.加载tokenizer
from transformers import BertTokenizer
token=BertTokenizer.from_pretrained('bert-base-chinese')



class Dataset(torch.utils.data.Dataset):
    def __init__(self, split):
        dataset= load_dataset(
    'csv',  # 指定文件格式为 csv，即使文件后缀是 .txt，也按 CSV 格式解析
    data_files='D:\\learning\\deep_learning\\code\\huggingface\\ChnSentiCorp\\train.tsv',
    split='train',
    delimiter='\t',  # 如果文件是用 Tab 分隔的，改为适合的分隔符（如 ','、'\t'）
    column_names=['label', 'text']  # 指定列名，按文件内容调整
         )

        def f(data):
            return len(data['text'])>30

        self.dataset=dataset.filter(f)




    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        #这个只需要text，不需要label，因为是一个填空问题
        text=self.dataset[idx]['text']
       # label = int(self.dataset[idx]['label'])
        return text

dataset=Dataset('train')

#print(dataset)
#print(dataset[0])


#定义数据的编码方式，将text转化为number,这才是正常的encoding方式，而不是直接使用python从底层开始encoding，这实在是太麻烦了
def collate_fn(data):
    data=token.batch_encode_plus(
        batch_text_or_text_pairs=data,
        padding='max_length',
        max_length=30,
        return_tensors='pt',
        return_length=True
    )

    input_ids=data['input_ids']
    attention_mask=data['attention_mask']
    token_type_ids=data['token_type_ids']

    #把第15个词替换为mask
    labels=input_ids[:,15].reshape(-1).clone()
    input_ids[:,15]=token.get_vocab()[token.mask_token]


    return input_ids, attention_mask, token_type_ids,labels


