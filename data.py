import os
import random
from settings import vocab
import torch

tag2id = {tag: idx for idx, tag in enumerate(vocab)}
id2tag = {idx: tag for idx, tag in enumerate(vocab)}

class Dataset:
    # 预测的时候传入的是单个文件，训练的时候是批量传入
    def __init__(self, file_folder_or_name, predict=False):
        super().__init__()
        self.sents = []
        self.tags = []
        self._file_size = []
        # 将多文件全部读入
        if not predict:
            file_name_list = [os.path.join(file_folder_or_name, name) for name in os.listdir(file_folder_or_name)]
        else:
            file_name_list = [file_folder_or_name]
        for file_name in file_name_list:
            with open(file_name, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                sum = 0
                for line in lines:
                    t = line.split('^')
                    if len(t)==2 and t[1].rstrip('\n') in vocab and not predict: # 确保格式正确
                        self.sents.append(t[0])
                        self.tags.append(t[1].rstrip('\n'))
                        sum += 1
                    elif predict: # 预测时不需要标签
                        self.sents.append(t[0].rstrip('\n'))
                self._file_size.append(sum) # 将每份文章的长度记录下来

    # def __getitem__(self, idx):
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
        # return self.sents[idx], tag2id[self.tags[idx]]

    # def __len__(self):
    #     return len(self.sents)

# 分文章返回sts和tag_ids的batch，而不是固定batch的大小
class DataIterator(Dataset):
    def __init__(self, file_folder):
        super().__init__(file_folder)
        self._file_size_copy = self._file_size.copy()
        self._all_size = 0

    def __len__(self):
        return len(self._file_size)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if len(self._file_size) != 0:
            current_size = self._file_size.pop(0)
            sts = self.sents[self._all_size: self._all_size + current_size]
            tag_ids = torch.tensor([tag2id[i] for i in self.tags[self._all_size: self._all_size + current_size]])
            self._all_size += current_size
            return sts, tag_ids
        else:
            self._all_size = 0 # 重置
            self._file_size = self._file_size_copy.copy()
            raise StopIteration


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
