import os
import random
from torch.utils.data import Dataset
from settings import vocab

tag2id = {tag: idx for idx, tag in enumerate(vocab)}
id2tag = {idx: tag for idx, tag in enumerate(vocab)}

class ProcessDataset(Dataset):
    def __init__(self, file_folder):
        super().__init__()
        self.sents = []
        self.tags = []
        # 将多文件全部读入
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


# 将data/label下的文件随机打乱并分成训练集和测试集
def data_divi():
    train_size = 0.8
    base_path = 'data/label/'

    # 读取data下的所有txt文件
    file_list = os.listdir(base_path)
    file_list = [file for file in file_list if file.endswith('.txt')]
    random.shuffle(file_list)

    # 分割训练集和测试集
    train_file_list = file_list[:int(len(file_list) * train_size)]
    test_file_list = file_list[int(len(file_list) * train_size):]

    # 分别移动
    for file in train_file_list:
        os.rename(base_path + file, base_path + 'train/' + file)
    for file in test_file_list:
        os.rename(base_path + file, base_path + 'test/' + file)
