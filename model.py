# from transformers import BertModel, BertTokenizer
import torch
from torch import nn
from text2vec import SentenceModel # 句嵌入模型
from settings import text2vec_model, vocab
from torchcrf import CRF

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
    def forward(self, sentences, tags=None, test=False):
        with torch.no_grad():
            embeds = self.text2vec_model.encode(sentences, convert_to_tensor=True)  # (batch_size, 768) 使用句嵌入模型
            #  embeds = torch.mean(self.bert(input_ids, attention_mask=attention_mask).last_hidden_state, dim=1) # (batch_size, sentence_len, 768) -> (batch_size, 768)
        enc, _ = self.bilstm(embeds)
        enc = self.dropout(enc)
        outputs = self.linear(enc)  # (batch_size, num_tags)
        outputs = outputs.unsqueeze(1)  # (batch_size, 1, num_tags)
        if not test:
            tags = tags.unsqueeze(1)  # (batch_size, 1, 1)
            loss = -self.crf.forward(outputs, tags, reduction='mean')
            return loss
        else:
            return self.crf.decode(outputs)  # (batch_size, 1, 1) 最后一个维度指的是tag值