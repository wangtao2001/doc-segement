# 实体类型
vocab = [
    'O',
    'S-CHAPTER',  # 章
    'S-SECTION',  # 节
    'S-ARTICLE',  # 条
    'B-ARTICLE',
    'I-ARTICLE',
    'E-ARTICLE',
]

text2vec_model = {
    'local': '/home/wangtao/models/text2vec-base-chinese',
    'remote': 'shibing624/text2vec-base-chinese'
}