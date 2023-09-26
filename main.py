import torch
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
import os
from model import Bert_BiLSTM_CRF
from data import DataIterator
import matplotlib.pyplot as plt
from run import train, test

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 100
epochs = 8
model = Bert_BiLSTM_CRF().to(device)
# 使用新的迭代器
train_iterator = DataIterator('data/label/train')
test_iterator = DataIterator('data/label/test')
optimizer = AdamW(model.parameters(), lr=1e-3, eps=1e-6)

total_steps = len(train_iterator) * epochs # 总步数
warm_up_ratio = 0.1 # 预热10%
# 线性学习率预热
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps*warm_up_ratio, num_training_steps=total_steps)

all_loss_list, all_lr_list = [], []
for epoch in range(epochs):
    all_loss, all_lr = train(epoch, model, train_iterator, optimizer, scheduler, device)
    # 记录训练过程中的信息
    all_loss_list.extend(all_loss)
    all_lr_list.extend(all_lr)
    # 由于缺乏验证集，所以每个epoch都进行测试查看模型性能
    test(epoch, model, test_iterator, device)

# 保存模型
torch.save(model, 'models/model.pt')

# 绘制loss曲线
plt.plot(all_loss_list)
plt.xlabel('step')
plt.ylabel('loss')
plt.title('loss curve')
plt.savefig('img/loss.png')
plt.close()
# 绘制lr曲线
plt.plot(all_lr_list)
plt.xlabel('step')
plt.ylabel('lr')
plt.title('lr curve')
plt.savefig('img/lr.png')
plt.close()