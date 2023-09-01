# 将data/label下的文件随机打乱并分成训练集和测试集
import os
import random

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


