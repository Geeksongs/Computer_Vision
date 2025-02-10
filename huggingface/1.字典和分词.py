from transformers import BertTokenizer

#1.这个是用来分词的工具
tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path='bert-base-chinese',
                                          cache_dir='None',
                                          force_download=False,)

#2.输入语料
sents=[
    '选择珠江花园的原因就是方便。',
    '笔记本的键盘确实爽',
    '我就是一个条狗，而你不是',
    '今天才知道这本书还有第六卷',
    '机器背面似乎有什么标签'
]

print(tokenizer,sents)

#3。现在进行简单的编码

out=tokenizer.encode(
    text=sents[0],
    text_pair=sents[1],


    truncation=True,
    padding='max_length',
    add_special_tokens=True,
    max_length=30,
    return_tensors=None,
)
print()
print(out)

out=tokenizer.decode(out)
print(out)

#4.现在进行批量编码
print("现在开始批量编码")
out=tokenizer.batch_encode_plus(
    batch_text_or_text_pairs=[sents[0],sents[1]],
    add_special_tokens=True,

    truncation=True,
    padding='max_length',
    max_length=15,

    return_tensors=None,
    return_token_type_ids=True,
    return_attention_mask=True,
    return_special_tokens_mask=True,

    return_length=True,
)

for k,v in out.items():
    print(k,':',v)

#开始我们已经对token进行了encode，现在开始了decode，也就是把
print(tokenizer.decode(out['input_ids'][0]))




