import torch
from datasets import load_dataset

#1.加载数据集
class Dataset(torch.utils.data.Dataset):
    def __init__(self, split):
        self.dataset= load_dataset(
    'csv',  # 指定文件格式为 csv，即使文件后缀是 .txt，也按 CSV 格式解析
    data_files='D:\\learning\\deep_learning\\code\\huggingface\\ChnSentiCorp\\train.tsv',
    split='train',
    delimiter='\t',  # 如果文件是用 Tab 分隔的，改为适合的分隔符（如 ','、'\t'）
    column_names=['label', 'text']  # 指定列名，按文件内容调整
)
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text=self.dataset[idx]['text']
        label=self.dataset[idx]['label']
       # label = int(self.dataset[idx]['label'])
        return text, label

dataset = Dataset('train')

print(len(dataset),dataset[0])
print(dataset[1])
print(dataset[2])
print("第一步完成！")
#2.加载tokenizer
from transformers import BertTokenizer
token=BertTokenizer.from_pretrained('bert-base-chinese')

# 3.定义批处理函数，这样就可以对成批的数据进行预处理



def collate_fn(data):
    filtered_data = []
    for sent, label in data:
        try:
            label = int(label)  # 尝试将 label 转换为整数
            filtered_data.append((sent, label))
        except ValueError:
            continue  # 如果无法转换，跳过该样本

    # 如果没有合法样本，返回空张量
    if not filtered_data:
        return torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0)

    sents = [i[0] for i in filtered_data]
    labels = [i[1] for i in filtered_data]

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

print("定义dataloader")
# 定义dataloader，这里我们已经自定义了batch size就是16
loader=torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=16,
    collate_fn=collate_fn,
    shuffle=True,
    drop_last=True,
)
print("开始循环")
for i,(input_ids,attention_mask,token_type_ids,labels) in enumerate(loader):
    break
print(len(loader))
print(input_ids.shape, attention_mask.shape, token_type_ids.shape,labels)

#加载预训练模型
from transformers import BertModel

print("开始加载预训练模型")
pretrained=BertModel.from_pretrained('bert-base-chinese')

for param in pretrained.parameters():
    param.requires_grad=False

#模型计算
print("开始模型计算前的检查")
print(len(input_ids))
print(len(attention_mask))
print(len(token_type_ids))
out=pretrained(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)


print("打印当前最后一个hidden layer的维度",out.last_hidden_state.shape)

#定义下游任务，这个是整个代码的关键，必须要弄明白每一个输入输出的维度！
class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        #因为bert的hidden size是768，这个是定死了的，对于bert-base-chinese这个模型
        self.fc=torch.nn.Linear(in_features=768,out_features=2)

    def forward(self,input_ids,attention_mask,token_type_ids):
        with torch.no_grad():
            out=pretrained(input_ids,attention_mask,token_type_ids)

        # last hidden state shape：torch.Size([16, 500, 768])
        # 对应的含义是：([batch_size, sequence_length, hidden_size]，相当于一个字对应一个hidden size的向量
        # 这里因为我设置了sequence_length=500，所以是500
        # last hidden state应该是bert的最后一层的输出
        # 索引【：，0】代表的是取出所有batch的第0个维度
        #print(out.last_hidden_state[:, 0])
        out=self.fc(out.last_hidden_state[:,0]) #应该是这个

        #然后现在out的shape应该变为了【16，1，768】,但是现在又经过了一层mlp，因此会变成【16,2】
        #print("shape1",out.shape)
        out=out.softmax(dim=1) #第一个维度是batch，后面的维度是1，因此是1
        #print("shape2", out.shape)
        return out

model=Model()
model(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids).shape


print("start the trainning process")

from transformers import AdamW

optimizer=AdamW(model.parameters(),lr=5e-4)

criterion=torch.nn.CrossEntropyLoss() #这个已经自带了softmax的操作，那么我们在模型里面其实不需要再写softmax了

# 读取测试数据集,由于测试数据集没有label，因此我们就直接使用dev数据集了。
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

dataset = load_dataset(
    'csv',  # 指定文件格式为 csv，即使文件后缀是 .txt，也按 CSV 格式解析
    data_files='D:\\learning\\deep_learning\\code\\huggingface\\ChnSentiCorp\\dev_modified.tsv',
    delimiter='\t',  # 如果文件是用 Tab 分隔的，改为适合的分隔符（如 ','、'\t'）
    column_names=['label', 'text'] , # 指定列名，按文件内容调整
    split='train'
)


subset_dataset = dataset.select(range(300))  # 选择前 300 个样本

loader_test = torch.utils.data.DataLoader(

    dataset=subset_dataset,
    batch_size=32,
    collate_fn=collate_fn_test,
    shuffle=True,
    drop_last=True,
)



#等下开始做做实验看看应该是不用再写了

correct = 0
total = 0

for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(loader):
    model.train()  # 切换到训练模式
    out = model(input_ids, attention_mask, token_type_ids)
    loss = criterion(out, labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # 每5个batch打印训练集上的准确率
    if i % 20 == 0:
        preds = out.argmax(dim=1)
        accuracy = (preds == labels).sum().item() / len(labels)
        print("The Iteration is {}, the loss is {:.4f}, the acc is {:.4f}".format(i, loss.item(), accuracy))

        # 使用测试集评估模型性能
        model.eval()  # 切换到评估模式
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for test_input_ids, test_attention_mask, test_token_type_ids, test_labels in loader_test:
                test_out = model(test_input_ids, test_attention_mask, test_token_type_ids)
                test_preds = test_out.argmax(dim=1)
                test_correct += (test_preds == test_labels).sum().item()
                test_total += len(test_labels)

        # 避免除以 0 的问题，是的确实需要这样搞
        if test_total > 0:
            test_accuracy = test_correct / test_total
            print("Test accuracy after {} iterations: {:.4f}".format(i, test_accuracy))
        else:
            print("Test loader contains no data. Please check loader_test.")

    # 限制训练到300个batch以减少训练时间（仅供测试）
    if i == 300:
        break










