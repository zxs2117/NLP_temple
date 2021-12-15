import torch
import torch.nn as nn
import torch.nn.functional as F

# config
class Config(object):
    def __init__(self,vocab_size,embed_size,target_size):
        self.model_name='textCNN'
        self.vocab_size=vocab_size
        self.embed_size=embed_size
        self.target_size=target_size
        self.embedding_pretrained=None
        self.filter_size=(2,3,4)
        self.num_filters=256
        self.dropout=0.2
        self.lr=1e-3


class textCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.embedding_pretrained is not None:
            self.embedding=nn.Embedding.from_pretrained(config.embedding_pretrained,freeze=False)
        else:
            self.embedding=nn.Embedding(config.vocab_size,config.embed_size,padding_idx=config.vocab_size-1)
        self.convs=nn.ModuleList(
                [nn.Conv2d(1,config.num_filters,(k,config.embed_size)) for k in config.filter_size]
            )
        self.dropout=nn.Dropout(config.dropout)
        self.linear=nn.Linear(config.num_filters*len(config.filter_size),config.target_size)

    def conv_and_pool(self,x,conv):
        x=F.relu(conv(x)).squeeze(3)
        x=F.max_pool1d(x,x.size(2)).squeeze(2)
        return x

    def forward(self,x):
        x=self.embedding(x).unsqueeze(1)
        x=torch.cat([self.conv_and_pool(x,conv) for conv in self.convs],1) #256*3
        x=self.dropout(x)
        x=self.linear(x)
        return x