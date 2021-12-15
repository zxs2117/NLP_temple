import textCNN
import bank_data_process
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
import torch


def train(model,epoch,opt,dataloader):
    model.train()

    for i in range(epoch):
        print("_______________第{}轮训练___________________\n".format(i + 1))
        index=0
        for datas,labels in tqdm(dataloader):
            model.zero_grad()
            output=model(datas)
            loss=F.cross_entropy(output,labels)
            loss.backward()
            opt.step()
            index+=1
            if index%100 ==0:
                true=labels.data.cpu()
                predic=torch.max(output.data,1)[1].cpu()
                train_acc=accuracy_score(true,predic)
                print('\n epoch:{} batch :{},loss={},train_acc:{}\n'.format(i+1,index,loss,train_acc))
    #save
    torch.save(model.state_dict(),'model/textCNN_1.pt')

    return model


def evalute(model,dataloader):
    model.eval()
    predict=[]
    label=[]
    with torch.no_grad():
        for datas,labels in dataloader:
            output=model(datas)
            predict+=torch.max(output.data,1)[1].cpu()
            label+=labels
    acc=accuracy_score(label,predict)
    return acc



if __name__=='__main__':


    #参数
    Embed_dim=20
    max_length=32
    batch_size=2
    epoch=5
    target_size=3


    #读取数据
    x,y=bank_data_process.get_data('../../data/bank_product/train_data_public.csv')
    vocab=bank_data_process.build_vocab(x)
    x,y=bank_data_process.data_process(x,vocab,labels=y)

    #切分数据
    x_train,x_dev,y_train,y_dev=train_test_split(x,y,test_size=0.2)

    #生成dataloader
    dataset_train=bank_data_process.TrainData(x_train,y_train)
    train_loader=DataLoader(dataset_train,batch_size=batch_size,shuffle=True)

    dataset_dev = bank_data_process.TrainData(x_dev, y_dev)
    dev_loader = DataLoader(dataset_dev, batch_size=batch_size, shuffle=True)

    #model and optimizier
    config=textCNN.Config(vocab_size=len(vocab),embed_size=Embed_dim,target_size=target_size)
    model=textCNN.textCNN(config)
    opt=Adam(model.parameters(),lr=config.lr)

    #train
    model=train(model,epoch,opt,train_loader)
    acc=evalute(model,dev_loader)

    print(acc)