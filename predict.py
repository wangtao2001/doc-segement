from data import Dataset, id2tag
from loader import doc2text
import torch
import os
import pandas as pd

doc_path = 'data/predict/doc'
txt_path = 'data/predict/txt'
result_path = 'data/predict/result'
result_txt_path = 'data/predict/result/txt'

def doc_predict():
    # 将文档转换为txt
    doc_list = os.listdir(doc_path)

    for file in doc_list:
        doc2text(os.path.join(doc_path, file), txt_path)

    model = torch.load('models/model.pt')

    # 预测并保存txt，单个文件单独执行
    txt_list = os.listdir(txt_path)
    for txt in txt_list:
        d = Dataset(os.path.join(txt_path, txt), predict=True)
        y_hat = [id2tag[y[0]] for y in model(d.sents, predict=True)]
        sts = d.sents

        # 这里做一步后处理，连续多个E出现，就把后面的E全部替换为S，否则在txt2excel中后面的E会被丢弃
        flag = False
        for i in range(len(y_hat)):
            if not flag and y_hat[i] == 'E-ARTICLE':
                flag = True
            elif flag and y_hat[i] == 'E-ARTICLE':
                y_hat[i] = 'S-ARTICLE'
            else:
                flag = False

        for i in range(len(sts)):
            sts[i] += '^' + y_hat[i] + '\n'
        # 保存一份txt
        with open(os.path.join(result_txt_path, txt), 'w' ,encoding='UTF-8') as f:
            f.writelines(sts)

# 将结果txt转换为excel
# 不在上面执行的原因是这里还可以手动对txt做修改
def txt2excel():
    txt_list = os.listdir(result_txt_path)
    for txt_name in txt_list:
        with open(os.path.join(result_txt_path, txt_name) , 'r', encoding='UTF-8') as f:
            sts = []
            tags = []
            lines = f.readlines()
            for line in lines:
                t = line.split('^')
                sts.append(t[0])
                tags.append(t[1].rstrip('\n'))
        current_chapter = ''
        current_chapter_id = 0
        current_section = ''
        current_section_id = 0
        current_article = ''
        current_article_id = 0
        current_id = 0
        data = [] # 待创建为df
        for i in range(len(sts)):
            write = False
            if tags[i] == 'O':
                continue
            elif tags[i] == 'S-CHAPTER':
                current_chapter = sts[i]
                current_chapter_id += 1
                # 章换了，节清空
                current_section = ''
                current_section_id = 0
            elif tags[i] == 'S-SECTION':
                current_section = sts[i]
                current_section_id += 1
            elif tags[i] == 'S-ARTICLE':
                current_article = sts[i]
                current_article_id += 1
                write = True
            elif tags[i] == 'B-ARTICLE':
                current_article = sts[i]
                current_article_id += 1
            elif tags[i] == 'E-ARTICLE':
                current_article += sts[i]
                write = True
            else: # I-ARTICLE
                current_article += sts[i]+ '\\\n'
            if write:
                current_id += 1
                c = str(current_chapter_id).zfill(2)
                s = str(current_section_id).zfill(2)
                a = str(current_article_id).zfill(3)
                data.append(['N' + str(current_id).zfill(4), 'T01', 'C01'+c if current_chapter_id != 0 else '', current_chapter, 'S01'+c+s if current_section_id else '', current_section, 'A01' +c+s+a, current_article])
        df = pd.DataFrame(data,columns=['id','title_id','chapter_id','chapter','section_id','section','article_id','article'], index=None)
        df.to_excel(os.path.join(result_path, f"{txt_name.strip('.txt')}.xlsx"), index=False)
            
if __name__ == "__main__":
    doc_predict()
    txt2excel()