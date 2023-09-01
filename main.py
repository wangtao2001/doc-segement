import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torchcrf import CRF
# from transformers import BertModel, BertTokenizer
from sklearn.metrics import accuracy_score
from text2vec import SentenceModel # 句嵌入模型
from settings import vocab, text2vec_model
from tqdm import tqdm
import os

tag2id = {tag: idx for idx, tag in enumerate(vocab)}
id2tag = {idx: tag for idx, tag in enumerate(vocab)}

# bert_model = 'bert-base-chinese'
# tokenizer = BertTokenizer.from_pretrained(bert_model)
# sentence_len = 100

class Bert_BiLSTM_CRF(nn.Module):
    def __init__(self, embedding_dim=768, hidden_dim=256, dropout=0.1, num_tags=len(vocab)):
        super().__init__()
        # 使用本地模型
        self.text2vec_model = SentenceModel(text2vec_model['local'])
        self.bilstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=2, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_dim, num_tags)
        self.crf = CRF(num_tags, batch_first=True)

    # def forward(self, input_ids, attention_mask, tag, test=False):
    def forward(self, sentences, tags, test=False):
        with torch.no_grad():
            embeds = self.text2vec_model.encode(sentences, convert_to_tensor=True)  # (batch_size, 768) 使用句嵌入模型
            #  embeds = torch.mean(self.bert(input_ids, attention_mask=attention_mask).last_hidden_state, dim=1) # (batch_size, sentence_len, 768) -> (batch_size, 768)
        enc, _ = self.bilstm(embeds)
        enc = self.dropout(enc)
        outputs = self.linear(enc)  # (batch_size, num_tags)
        outputs = outputs.unsqueeze(1)  # (batch_size, 1, num_tags)
        tags = tags.unsqueeze(1)  # (batch_size, 1, 1)
        if not test:
            loss = -self.crf.forward(outputs, tags, reduction='mean')
            return loss
        else:
            return self.crf.decode(outputs)  # (batch_size, 1, 1) 最后一个维度指的是tag值


class ProcessDataset(Dataset):
    def __init__(self, file_folder):
        super().__init__()
        self.sents = []
        self.tags = []
        # 支持多文件使用
        for file_name in os.listdir(file_folder):
            with open(os.path.join(file_folder, file_name), 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    t = line.split('^')
                    if len(t)==2 and t[1].rstrip('\n') in vocab: # 确保格式正确
                        self.sents.append(t[0])
                        self.tags.append(t[1].rstrip('\n'))

    def __getitem__(self, idx):
        # token = tokenizer.encode_plus(
        #         text=self.sents[idx], 
        #         truncation=True,
        #         padding='max_length',
        #         max_length=sentence_len,
        #         add_special_tokens=True,
        #         return_attention_mask=True,
        #     )
        # input_ids = torch.tensor(token['input_ids'])
        # attention_mask = torch.tensor(token['attention_mask'])
        # return input_ids, attention_mask, tag2id[self.tags[idx]]
        return self.sents[idx], tag2id[self.tags[idx]]

    def __len__(self):
        return len(self.sents)


def train(epoch, model, iterator, optimizer, device):
    model.train()
    losses = 0.0
    step = 0
    with tqdm(total=len(iterator)) as pbar:
        for sts, tag_ids in iterator:
            step += 1
            pbar.update(1)
            # sts 是一个tuple
            tag_ids = tag_ids.to(device)

            loss = model(sts, tag_ids)
            losses += loss.item()  # 返回loss的标量值
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    print(f'epoch: {epoch}, loss:{losses / step}')  # 每个epoch的平均loss


def test(epoch, model, iterator, device):
    model.eval()
    Y, Y_hat = [], []
    losses = 0
    step = 0
    with tqdm(total=len(iterator)) as pbar:
        with torch.no_grad():
            for sts, tag_ids in iterator:
                step += 1
                pbar.update(1)
                tag_ids = tag_ids.to(device)

                y_hat = model(sts, tag_ids, test=True)

                loss = model(sts, tag_ids)  # 同时获取loss和预测值
                losses += loss.item()
                Y.append(tag_ids.tolist())  # tensor([tag, tag, tag, ...])
                Y_hat.append([i[0] for i in y_hat])  # [[tag], [tag], [tag], ...]

                # y_true = [id2tag[i] for i in Y]
                # y_pred = [id2tag[i] for i in Y_hat]

    print(Y_hat)
    print(Y)
    acc = (Y_hat == Y).mean()*100 #accuracy_score(Y, Y_hat) * 100
    print(f"epoch: {epoch}, test Loss:{losses / step}, test Acc:{acc}%")
    return model, losses / step, acc

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 20
model = Bert_BiLSTM_CRF().cuda()
train_dataset = ProcessDataset('data/label/train')
train_iterator = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = ProcessDataset('data/label/test')
test_iterator = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
optimizer = Adam(model.parameters(), lr=1e-3) # 关于学习率预热

for epoch in range(10):
    # train(epoch, model, train_iterator, optimizer, device)
    model, val_loss, val_acc = test(epoch, model, train_iterator, device)
    if val_acc > 90:
        # 保存模型
        torch.save(model, f'model/model-epoch:{epoch}.pt')
