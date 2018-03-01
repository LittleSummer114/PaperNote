#encoding:utf-8

import torch.nn as nn
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self,**kwargs):
        super(CNN,self).__init__()
        self.name='CNN'
        self.dimension = kwargs['dimension']
        self.batch_size = kwargs['batch_size']
        self.type = kwargs['type']
        self.classes=kwargs['classes']
        self.number_of_filters = kwargs['number_of_filters']
        self.filter_size = kwargs['filter_size']
        self.epoch=kwargs['epoch']
        self.wv = kwargs['wv_maritx']
        self.dropout_rate = kwargs['dropout']
        self.level = kwargs['level']
        self.VOCABULARY_SIZE = kwargs['VOCABULARY_'+self.level+'_SIZE']
        self.max_sent_length = kwargs['max_sent_'+self.level+'_length']
        self.length_feature = kwargs['length_feature']
        self.max_sent_length+=self.length_feature
        self.embedding=nn.Embedding(self.VOCABULARY_SIZE+2,self.dimension,padding_idx=self.VOCABULARY_SIZE+1)
        #self.embedding.weight=nn.Parameter(torch.LongTensor(self.wv))
        self.relu=nn.ReLU()
        self.channel=1
        if(self.type=='static'):
            self.embedding.weight.requires_grad = False
        if(self.type=='multichannel'):
            self.channel=2
        if(self.type=='non-static'):
            self.embedding.weight.data.copy_(torch.from_numpy(self.wv))
        self.dropout=nn.Dropout(p=self.dropout_rate)
        for i in range(len(self.number_of_filters)):
            con=nn.Conv1d(self.channel,self.number_of_filters[i],self.dimension*self.filter_size[i],stride=self.dimension)
            setattr(self,'con_{}'.format(i),con)
        self.fc=nn.Linear(sum(self.number_of_filters),len(self.classes))
        self.softmax=nn.Softmax(dim=1)

    def con(self,i):
        return getattr(self,'con_{}'.format(i))

    def forward(self,input):
        x = self.embedding(input).view(-1, 1, self.dimension * self.max_sent_length)
        conv=[]
        for i in range(len(self.number_of_filters)):
            temp=self.con(i)(x)
            temp=self.relu(temp)
            temp=nn.MaxPool1d(self.max_sent_length-self.filter_size[i]+1)(temp).view(-1,self.number_of_filters[i])
            conv.append(temp)
        x=torch.cat(conv,1)
        x = self.dropout(x)
        x = self.fc(x)
        #x = self.softmax(x)
        return x

class CNN2(nn.Module):
    def __init__(self,*kwargs):
        super(CNN2,self).__init__()
        self.dimension=kwargs['dimension']
        self.VOCABULARIZE_WORD=kwargs['VOCABULARY_WORD_SIZE']
        self.VOCABULARIZE_CHAR = kwargs['VOCABULARY_CHAR_SIZE']
        self.name = 'CNN'
        self.batch_size = kwargs['batch_size']
        self.type = kwargs['type']
        self.classes = kwargs['classes']
        self.number_of_filters = kwargs['number_of_filters']
        self.filter_size = kwargs['filter_size']
        self.epoch = kwargs['epoch']
        self.wv = kwargs['wv_maritx']
        self.dropout_rate = kwargs['dropout']
        self.max_sent_word_length = kwargs['max_sent_word_length']
        self.max_sent_char_length = kwargs['max_sent_char_length']
        self.embedding_word=nn.Embedding(self.VOCABULARIZE_WORD+2,self.dimension,padding_idx=self.VOCABULARIZE_WORD+1)
        self.embedding_word.weight.data._copy(torch.from_numpy(self.wv))

class TextRnn(nn.Module):
    def __init__(self,**kwargs):
        super(TextRnn,self).__init__()
        self.dimension = kwargs['dimension']
        self.VOCABULARIZE_WORD = kwargs['VOCABULARY_WORD_SIZE']
        self.VOCABULARIZE_CHAR = kwargs['VOCABULARY_CHAR_SIZE']
        self.name = 'CNN'
        self.dimension = kwargs['dimension']
        self.batch_size = kwargs['batch_size']
        self.type = kwargs['type']
        self.classes = kwargs['classes']
        self.number_of_filters = kwargs['number_of_filters']
        self.filter_size = kwargs['filter_size']
        self.epoch = kwargs['epoch']
        self.wv = kwargs['wv_maritx']
        self.dropout_rate = kwargs['dropout']
        self.max_sent_word_length = kwargs['max_sent_word_length']
        self.max_sent_char_length = kwargs['max_sent_char_length']
        self.embedding_word = nn.Embedding(self.VOCABULARIZE_WORD + 2, self.dimension,
                                           padding_idx=self.VOCABULARIZE_WORD + 1)
        self.embedding_word.weight.data._copy(torch.from_numpy(self.wv))
        self.lstm=nn.LSTM(num_layers=3)
    def forward(self):
        pass
