import pandas as pd

import textCNN
import bank_data_process
from torch.utils.data import DataLoader
import torch
import numpy as np


def predict(model,dataloader):
    model.eval()
    predict_all = np.array([], dtype=int)

    with torch.no_grad():
        for datas in dataloader:
            output=model(datas)
            predict=torch.max(output.data,1)[1].cpu()
            print(predict)
            predict_all = np.append(predict_all, predict)


    return predict_all








if __name__=='__main__':


    #参数
    Embed_dim=20
    max_length=32
    batch_size=1
    epoch=5
    target_size=3


    #读取数据
    x_text=bank_data_process.get_data('../../data/bank_product/test_public.csv',train=False)
    vocab=bank_data_process.build_vocab(x_text)
    x,_=bank_data_process.data_process(x_text,vocab)

    dataset_test = bank_data_process.TestData(x)
    test_loader = DataLoader(dataset_test, batch_size=batch_size)

    #model
    config=textCNN.Config(vocab_size=len(vocab),embed_size=Embed_dim,target_size=target_size)
    model=textCNN.textCNN(config)
    model.load_state_dict(torch.load('model/textCNN_1.pt'))

    result=predict(model,dataloader=test_loader)
    print(len(x))
    print(len(result))
    #save_result
    df=pd.DataFrame(columns=['text','class'])
    df['text']=x_text
    df['class']=result
    df.to_csv('result.csv')




