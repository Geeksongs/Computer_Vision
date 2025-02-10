from transformers import BertTokenizer

#1.这个是用来分词的工具
tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path='bert-base-chinese',
                                          cache_dir='None',
                                          force_download=False,)

#2.输入语料
sents=[
    '选择珠江花园的原因就是方便。',
    '笔记本的键盘确实爽',
    '今晚的月光很美丽，适合做爱'
    '我就是一个条狗，而你不是',
    '今天才知道这本书还有第六卷',
    '机器背面似乎有什么标签'
]

print(tokenizer,sents)

print()

diction=tokenizer.get_vocab()
print(type(diction),len(diction))
print('月光' in diction)
print('做爱' in diction)

# 添加新词和特殊符号
tokenizer.add_tokens(new_tokens=['月光','做爱'])
tokenizer.add_special_tokens({'eos_token':'[EOS]'})



#3.编码新词，开始做爱

out=tokenizer.encode(
    text='在月光下开始做爱',
    text_pair=None,

    truncation=True,
    padding='max_length',
    add_special_tokens=True,
    max_length=8,
    return_tensors=None,
)

print(out)

print(tokenizer.decode(out))
