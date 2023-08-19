# 实体类型
vocab = [
    'O',
    'S-TITLE', # 标题
    'S-CHAPTER', # 章
    'S-SECTION', # 节
    'S-ARTICLE', # 条
    'B-ARTICLE',
    'I-ARTICLE',
    'E-ARTICLE',
]

# 实体类型标签与id映射
tag2id = {tag: idx for idx, tag in enumerate(vocab)}
id2tag = {idx: tag for idx, tag in enumerate(vocab)}
