import json
from torch.utils.data import DataLoader,Dataset
import os
import jieba
import pandas as pd
import re
import torch
import numpy as np
from sklearn.model_selection import train_test_split

#从文件中读取数据,并预处理
def get_data(path,train=True):
    with open(path, 'r', encoding='utf-8') as f:
        data = pd.read_csv(f)
    x=data['text']
    #数据清洗
    x_clean=[]
    for i in x:
        i=i.strip()
        i=re.sub(r'[0-9a-zA-Z！@？。，；’：”<>.' ']',"",i)
        x_clean.append(i)
    #返回x,y
    if train:
         return x_clean,data['class']
    else:
         return x_clean


#构建词表
def build_vocab(data,path='D:\\Python\\NLP_temple\\classifier\\cnews\\vocab.json'):
    #load vocab
    if os.path.exists(path):
        with open(path, 'r') as f:
            word2idx = json.loads(f.read())
        return word2idx
    else:
        word2idx={}
        word2idx['PAD']=0
        word2idx['UNK']=1
        for setence in data:
            for word in jieba.cut(setence):
                if word not in word2idx:
                    word2idx[word]=len(word2idx)
        #保存vocab
        vocab_js = json.dumps(word2idx)
        with open('vocab.json', 'w') as json_f:
            json_f.write(vocab_js)
        return word2idx

#TrainData
class TrainData(Dataset):
    def __init__(self, x_train, y_train, device='cpu'):
        self.x_data = torch.from_numpy(x_train).long().to(device)
        self.y_data = torch.from_numpy(y_train).long().to(device)
        self.len = x_train.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

class TestData(Dataset):
    def __init__(self, x_train, device='cpu'):
        self.x_data = torch.from_numpy(x_train).long().to(device)
        self.len = x_train.shape[0]

    def __getitem__(self, index):
        return self.x_data[index]

    def __len__(self):
        return self.len

#对句子进行padding，word2idx
def data_process(data,vocab,target_vocab={},max_length=32,labels=[]):

    x=[]
    for sentence in data:
        words=[]
        for word in jieba.cut(sentence):
            #识别是否UNK
            if word not in vocab:
                words.append(vocab['UNK'])
            else:
                words.append(vocab[word])
        #进行Padding
        if len(words) >= max_length:
            words = words[:32]
        else:
            words += [vocab['PAD']] * (max_length - len(words))
        x.append(words)

    if target_vocab=={}:
        labels=[int(label) for label in labels]
    else:
        labels=[int(target_vocab[label]) for label in labels]
    x=np.array(x)
    y=np.array(labels)
    return x,y



if __name__=='__main__':
    x,y=get_data('../../data/bank_product/train_data_public.csv')
    vocab=build_vocab(x)
    print(vocab['UNK'])
    x,y=data_process(x,vocab,labels=y)
    print(x.shape)
    print(y.shape)
    x_train,x_dev,y_train,y_dev=train_test_split(x,y,test_size=0.2)
    print(x_train.shape)
    print(y_train.shape)
