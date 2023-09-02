from modelscope.pipelines import pipeline
import os
from random import choice
import zhon
import string
import pandas as pd

p = pipeline(
    task='document-segmentation',
    model='damo/nlp_bert_document-segmentation_chinese-base')


# 随机获取一个文件
base_path = 'data/label/train'
document = ''
file_name = choice(os.listdir(base_path))
with open(os.path.join(base_path, file_name)) as txt:
    lines = txt.readlines()
    for line in lines:
        line = line.split('^')[0]
        # 两种方式：一、将换行去掉，取之在句子之间添加句号 二、保留换行
        if line[-1] in zhon.hanzi.punctuation or line[-1] in string.punctuation:
            pass
        else:
            line += '。'
        # line += '\n'
        # document += line

result = p(document)
text_list = result['text'].split('\t')
text_list = [text.rstrip('\n') for text in text_list]

# 将text_list写入excel
df = pd.DataFrame(text_list)
df.to_excel(f'test-{file_name}.xlsx', index=False, header=False)