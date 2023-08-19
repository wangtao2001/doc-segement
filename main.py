import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torchcrf import CRF
# from transformers import BertModel, BertTokenizer
from sklearn.metrics import accuracy_score
from text2vec import SentenceModel
from settings import vocab, tag2id, id2tag

# bert_model = 'bert-base-chinese'
# tokenizer = BertTokenizer.from_pretrained(bert_model)
sentence_len = 30

class Bert_BiLSTM_CRF(nn.Module):
    def __init__(self, emdedding_dim=768, hidden_dim=256, dropout=0.1, num_tags=len(vocab)):
        super().__init__()
        self.text2vec_model = SentenceModel()
        self.bilstm = nn.LSTM(emdedding_dim, hidden_dim // 2, num_layers=2, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_dim, num_tags)
        self.crf = CRF(num_tags, batch_first=True)

    # def forward(self, input_ids, attention_mask, tag, test=False):
    def forward(self, sentences, tags, test=False):
        with torch.no_grad():
            embeds = torch.tensor(self.text2vec_model.encode(sentences)) # (batch_size, 768) 使用句嵌入模型
            print(embeds)
            # embeds = torch.mean(self.bert(input_ids, attention_mask=attention_mask).last_hidden_state, dim=1) # (batch_size, sentence_len, 768) -> (batch_size, 768)
        enc, _ = self.bilstm(embeds)
        enc = self.dropout(enc)
        outpus = self.linear(enc) # (batch_size, num_tags)
        outpus = outpus.unsqueeze(1) # (batch_size, 1, num_tags) 
        tags = tags.unsqueeze(1) # (batch_size, 1, 1)
        if not test:
            loss = -self.crf.forward(outpus, tags, reduction='mean')
            return loss
        else:
            return self.crf.decode(outpus) # (batch_size, 1, 1) 最后一个维度指的是tag值
    
class ProcessDataset(Dataset):
    def __init__(self, file_path):
        super().__init__()
        self.sents = []
        self.tags = []
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                self.sents.append(line.split('^')[0])
                self.tags.append(line.split('^')[1].rstrip('\n'))
        
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
    for sts, tag_ids in iterator:
        step += 1
        # sts 是一个tuple
        tag_ids = tag_ids.to(device)

        loss = model(sts, tag_ids)
        losses += loss.item() # 返回loss的标量值
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(f'epoch: {epoch}, loss:{losses/step}') # 每个epoch的平均loss

def validate(epoch, model, iterator, device):
    model.eval()
    Y, Y_hat = [], []
    losses = 0
    step = 0
    with torch.no_grad():
        for sts, tag_ids in iterator:
            step += 1
            # sts 是一个tuple
            tag_ids = tag_ids.to(device)

            y_hat = model(sts, tag_ids, test=True)

            loss = model(sts, tag_ids) # 同时获取loss和预测值
            losses += loss.item()
            Y.append(tag_ids.tolist()) # tensor([tag, tag, tag, ...])
            Y_hat.append([i[0] for i in y_hat])  # [[tag], [tag], [tag], ...]

    acc = accuracy_score(Y, Y_hat) * 100
    print(f"epoch: {epoch}, val Loss:{losses/step}, val Acc:{acc}%")
    return model, losses/step, acc

def test(model, iterator, device):
    model.eval()
    Y, Y_hat = [], []
    with torch.no_grad():
        for sts, tag_ids in iterator:
            step += 1
            tag_ids = tag_ids.to(device)

            y_hat = model(sts, tag_ids, test=True)

            Y.append(tag_ids.tolist())
            Y_hat.append([i[0] for i in y_hat])

    y_true = [id2tag[i] for i in Y]
    y_pred = [id2tag[i] for i in Y_hat]

    return y_true, y_pred



batch_size = 5
model = Bert_BiLSTM_CRF()
dataset = ProcessDataset('data/label/process1.txt')
train_iterator = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
optimizer = Adam(model.parameters(), lr=1e-3)
device = torch.device('cpu')

validate(1, model, train_iterator, device)
