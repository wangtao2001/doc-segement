from tqdm import tqdm
from seqeval.metrics import *
from seqeval.scheme import IOBES
import torch
from data import id2tag

def train(epoch, model, iterator, optimizer, scheduler, device):
    model.train()
    all_loss = []
    all_lr = []
    model.train()
    Y, Y_hat = [], []
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
            all_loss.append(loss.item()) # 记录所有的loss
            all_lr.append(optimizer.param_groups[0]['lr']) # 记录所有的lr
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            y_hat = model(sts, tag_ids,  predict=True)
            y_hat = [i[0] for i in y_hat] # [[1], [3], ...]
            y = tag_ids.tolist() # tensor(1, 3, ....)
            # 再转换为BIOES形式
            y_hat = [id2tag[i] for i in y_hat]
            y = [id2tag[i] for i in y]
            # 一个epoch之后再计算，防止实体被batch截断
            Y_hat.extend(y_hat)
            Y.extend(y)
    print(f"train mode: epoch:{epoch}")
    print("acc: ", accuracy_score([Y], [Y_hat])) # https://github.com/ibatra/BERT-Keyword-Extractor/issues/17
    # print("p: ", precision_score([Y], [Y_hat], mode='strict', scheme=IOBES))
    # print("r: ", recall_score([Y], [Y_hat], mode='strict', scheme=IOBES))
    # print("f1: ", f1_score([Y], [Y_hat], mode='strict', scheme=IOBES))
    print("classification report: ")
    print(classification_report([Y], [Y_hat], mode='strict', scheme=IOBES))
    print(f"train loss:{losses / step}")  # 每个epoch的平均loss
    return all_loss, all_lr


def test(epoch, model, iterator, device):
    model.eval() # 评估模式
    step = 0
    Y, Y_hat = [], []
    with tqdm(total=len(iterator)) as pbar:
        with torch.no_grad():
            for sts, tag_ids in iterator:
                step += 1
                pbar.update(1)
                tag_ids = tag_ids.to(device)

                y_hat = model(sts, predict=True)
                y_hat = [i[0] for i in y_hat]
                y = tag_ids.tolist()
                y_hat = [id2tag[i] for i in y_hat]
                y = [id2tag[i] for i in y]
                Y_hat.extend(y_hat)
                Y.extend(y)
    print(f"test mode: epoch:{epoch}")
    print("acc: ", accuracy_score([Y], [Y_hat]))
    print("classification report: ")
    print(classification_report([Y], [Y_hat], mode='strict', scheme=IOBES))