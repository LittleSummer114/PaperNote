#encoding:utf-8
# to process data x:['I','Like','eating','apples'] --> y[0]
#输入是一个数据集的名字
#输出是 数据集,训练集 测试集 验证集 还有词汇量 word_index index_word
import numpy as np
np.random.seed(7)
import pickle
import math
import pandas
from pandas import DataFrame,read_csv
import jieba
import os
import time
from sklearn.utils import shuffle
from gensim.models import KeyedVectors
from gensim.models.word2vec import Word2Vec
import matplotlib.pyplot as plt

# for neural-network based method
def Sen2Index(data,params):
    def char2idx(name):
        x = []
        y = []
        for sent in data[name+'_x']:
            sentence_char=[]
            len_chars = 0
            for word in sent:
                for char in word:
                    if(len(sentence_char) < params ['max_sent_char_length']):
                        len_chars += 1
                        sentence_char.append(data['char_to_idx'][char])
            sentence_char += [len(data['vocab_char'])+1]*(params['max_sent_char_length']-(len_chars))
            if(params['length_feature']==1):
                sentence_char.append(len_chars)
            x.append(sentence_char)
        for label in data[name+'_y']:
            if(label in data['classes']):
                y.append(data['classes'].index(label))
        return x,y
    def word2idx(name):
        x = []
        y = []
        for sent in data[name+'_x']:
            sentence_word = []
            for word in sent:
                sentence_word.append(data['word_to_idx'][word])
                if (len(sentence_word) == params['max_sent_word_length']):
                    break
            sentence_word += [(len(data['vocab_word']) + 1)] * (params['max_sent_word_length'] - len(sent))
            if (params['length_feature'] == 1):
                sentence_word.append(len(sent))
            x.append(sentence_word)
        for c in data[name+'_y']:
            if(c in data['classes']):
                y.append(data['classes'].index(c))
        return x,y
    data['train_x_word'],data['train_y_word'] = word2idx('train')
    data['dev_x_word'], data['dev_y_word'] = word2idx('dev')
    data['test_x_word'], data['test_y_word'] = word2idx('test')
    data['x_word'] = data['train_x_word'] + data ['test_x_word'] + data ['dev_x_word']
    data['y_word'] = data['train_y_word'] + data ['test_y_word'] + data ['dev_y_word']
    data['train_x_char'], data['train_y_char'] = char2idx('train')
    data['dev_x_char'], data['dev_y_char'] = char2idx('dev')
    data['test_x_char'], data['test_y_char'] = char2idx('test')
    data['x_char'] = data['train_x_char'] + data['test_x_char'] + data['dev_x_char']
    data['y_char'] = data['train_y_char'] + data['test_y_char'] + data['dev_y_char']
    return data

def clear_string(sent):
    sent = sent.replace('< br / > n', ' ')
    sent = sent.replace('<br />rn','')
    sent = sent.replace('& quot;', '')
    return sent

def getStopWords():
    stopwords={}
    data=open('data/CH_stopWords.txt',encoding='utf-8').readlines()
    for line in data:
        line=line.strip()
        stopwords[line]=1
    return stopwords

def SplitSentence(sent,stopwords,vocab):
    sent=clear_string(sent)
    string=''
    words=jieba.cut(sent)
    sentence=[]
    for word in words:
        if(stopwords.get(word)is None and vocab.get(word)is not None):
            sentence.append(word)
    count=0
    for word in sentence:
        count+=1
        string+=word
        if(count<len(sentence)):
            string+=' '
    return string

def read_TREC():
    data={}
    def read(name):
        filename='data/TREC/'+name+'.txt'
        data=open(filename,encoding='utf-8')
        x = []
        y = []
        Discuss = []
        for line in data:
            line=line.split('\t')
            y.append(line[0].split(':')[0])
            x.append(line[1].split())
            Discuss.append(line[1])
        return x,y,Discuss
    train_x,train_y,train_Discuss=read('train')
    train_x,train_y,train_Discuss = shuffle(train_x,train_y,train_Discuss)
    test_x,test_y,test_Discuss=read('test')
    train_x_index=len(train_x)
    dev_x_index=train_x_index//10
    data['train_x'] = train_x[dev_x_index:train_x_index]
    data['dev_x']=train_x[:dev_x_index]
    data['train_Discuss'] = train_Discuss[dev_x_index:train_x_index]
    data['dev_Discuss'] = train_Discuss[:dev_x_index]
    data['train_y'] = train_y[dev_x_index:train_x_index]
    data['dev_y'] = train_y[:dev_x_index]
    data['test_x']=test_x
    data['test_y']=test_y
    data['test_Discuss'] = test_Discuss
    return data

def read_MR():
    data={}
    def read(name):
        filename='data/MR/'+name+'.txt'
        data=open(filename,encoding='utf-8').readlines()
        x=[]
        y=[]
        Discuss = []
        for line in data:
            line=line.strip().split('\t')
            y.append(line[0])
            x.append(line[1].split())
            Discuss.append(line[1])
        return x,y,Discuss
    train_x, train_y, train_Discuss = read('train')
    train_x, train_y, train_Discuss =shuffle(train_x,train_y,train_Discuss)
    test_x, test_y, test_Discuss =read('test')
    train_x_index=len(train_x)
    dev_x_index=train_x_index//10
    data['train_x'] = train_x[dev_x_index:train_x_index]
    data['dev_x']=train_x[:dev_x_index]
    data['train_Discuss'] = train_Discuss[dev_x_index:train_x_index]
    data['dev_Discuss'] = train_Discuss[:dev_x_index]
    data['train_y'] = train_y[dev_x_index:train_x_index]
    data['dev_y'] = train_y[:dev_x_index]
    data['test_x']=test_x
    data['test_y']=test_y
    data['test_Discuss'] = test_Discuss
    return data

def list2dic(temp_list):
    temp_dic={}
    for key in temp_list:
        temp_dic[key]=1
    return temp_dic

def read_Travel(data):
    stopwords=getStopWords()
    vocab=list2dic(data['vocab_word'])
    def read(name):
        x=[]
        y=[]
        Discuss=[]
        filename = 'data/Travel/' + name + '.csv'
        Travel_data = read_csv(filename, encoding='utf-8')
        if(name=='train'):
            Travel_data = Travel_data.drop_duplicates(['Discuss'])
        result = DataFrame()
        for idx,content in Travel_data.iterrows():
            string=SplitSentence(content['Discuss'],stopwords,vocab)
            Discuss.append(string)
            x.append(string.split())
            if (name == 'test'):
                y.append('NULL')
            else:
                y.append(content['Score'])
        result['Id'] = Travel_data['Id']
        result['Discuss'] = Discuss
        result['Score'] = y
        return x, y,result['Id'],result['Discuss']
    time1 = time.time()
    train_x,train_y ,train_id,train_Discuss=read('train')
    test_x,test_y,test_id,test_Discuss=read('test')
    train_x_index = len(train_x)
    dev_x_index = train_x_index // 10
    train_x,train_y = train_x,train_y
    train_x,train_y = shuffle(train_x,train_y)
    data['train_x'] = train_x[dev_x_index:train_x_index]
    data['dev_x'] = train_x[:dev_x_index]
    data['train_y'] = train_y[dev_x_index:train_x_index]
    data['dev_y'] = train_y[:dev_x_index]
    data['test_x'] = test_x
    data['test_y'] = test_y
    data['train_Id'], data['train_Discuss'] = train_id[dev_x_index:train_x_index], train_Discuss[dev_x_index:train_x_index]
    data['dev_Id'], data['dev_Discuss'] = train_id[:dev_x_index], train_Discuss[:dev_x_index]
    data['test_Id'], data['test_Discuss'] = test_id, test_Discuss
    time2 = time.time()
    print('Load Dataset Time:', str(time2 - time1))
    return data

def dataAugument():
    '''
    数据增强,将类别比较少的数据两两组合
    :return:
    '''
def read_TravelTest():
    data={}
    stopwords=getStopWords()
    def read(name):
        x=[]
        y=[]
        Discuss=[]
        filename = 'data/TravelTest/' + name + '.csv'
        resultname = 'data/TravelTest/' + name + '_split.csv'
        Travel_data = read_csv(filename,encoding='utf-8')
        result = DataFrame()
        for i in range(len(Travel_data['Id'])):
            string=SplitSentence(Travel_data['Discuss'][i],stopwords)
            Discuss.append(string)
            x.append(string.split())
            if(name=='test'):
                y.append('NULL')
            else:
                y.append(Travel_data['Score'][i])
        result['Id']=Travel_data['Id']
        result['Discuss']=Travel_data['Discuss']
        result['Score']=y
        result.to_csv(resultname,encoding='utf-8',index=False)
        return x, y,Travel_data['Id'],Travel_data['Discuss']
    train_x,train_y,train_id,train_Discuss=read('train')
    test_x,test_y,test_id,test_Discuss=read('test')
    train_x_index = len(train_x)
    dev_x_index = train_x_index // 10
    train_x, train_y = train_x, train_y
    train_x, train_y = shuffle(train_x, train_y)
    data['train_x'] = train_x[dev_x_index:train_x_index]
    data['dev_x'] = train_x[:dev_x_index]
    data['train_y'] = train_y[dev_x_index:train_x_index]
    data['dev_y'] = train_y[:dev_x_index]
    data['test_x'] = test_x
    data['test_y'] = test_y
    data['train_Id'], data['train_Discuss'] = train_id[dev_x_index:train_x_index], train_Discuss[
                                                                                   dev_x_index:train_x_index]
    data['dev_Id'], data['dev_Discuss'] = train_id[:dev_x_index], train_Discuss[:dev_x_index]
    data['test_Id'], data['test_Discuss'] = test_id, test_Discuss
    return data

def getVocab(params):
    vocab_word_file = 'data/' + params['dataset'] + '/vocab_word.csv'
    vocab_char_file = 'data/' + params['dataset'] + '/vocab_char.csv'
    if (os.path.exists(vocab_word_file) and os.path.exists(vocab_char_file)):
        vocab_word = read_csv(vocab_word_file, encoding='utf-8')['vocab_word'].tolist()
        vocab_char = read_csv(vocab_char_file, encoding='utf-8')['vocab_char'].tolist()
    else:
        data = eval('read_{}'.format(params['dataset']))()
        data['x'] = data['train_x'] + data['dev_x'] + data['test_x']
        data['y'] = data['train_y'] + data['test_y'] + data['dev_y']
        vocab_word = []
        vocab_char = []
        for sent in data['x']:
            for word in sent:
                vocab_word.append(word)
                for char in word:
                    vocab_char.append(char)
        vocab_word = list(set(vocab_word))
        vocab_char = list(set(vocab_char))
        temp_vocab_word = DataFrame()
        temp_vocab_word['vocab_word'] = vocab_word
        temp_vocab_word.to_csv(vocab_word_file, encoding='utf-8', index=False)
        temp_vocab_char = DataFrame()
        temp_vocab_char['vocab_char'] = vocab_char
        temp_vocab_char.to_csv(vocab_char_file, encoding='utf-8', index=False)
    return vocab_word, vocab_char

def getMaxLength(sentences):
    len_word=[]
    len_char=[]
    for sent in sentences:
        len_word.append(len(sent))
        temp_len_char=0
        for word in sent:
            temp_len_char+=len(word)
        len_char.append(temp_len_char)
    return len_word,len_char

def getHit(train,test,params,name):
    sent_word_file = 'data/' + params['dataset'] + '/' + 'sent_'+name+'_dis.eps'
    plt.clf()
    plt.figure(1)
    plt.title('sentence_length_'+name)
    plt.xlabel('Length')
    plt.ylabel('Count')
    plt.hist(train,bins=100, label='train', lw=1)
    plt.hist(test, bins=100,label='test', lw=1)
    plt.legend(loc='upper right')
    plt.figure(1).savefig(sent_word_file)
    plt.close()

def DataAnalysis(data,params):
    train_word, train_char = getMaxLength(data['train_x'] + data['dev_x'])
    test_word, test_char = getMaxLength(data['test_x'])
    getHit(train_word, test_word, params, 'word')
    getHit(train_char, test_char, params, 'char')
    def getMax(name,all):
        print('max_sent_'+name+'_length', max(all))
        print('average_sent_'+name+'_length', np.average(np.array(all)))
    getMax('word',train_word+test_word)
    getMax('char',train_char+test_char)

def element2idx(data,name):
    data['idx_to_'+name] = {}
    data[name+'_to_idx'] = {}
    for key, word in enumerate(data['vocab_'+name]):
        data['idx_to_'+name][key] = word
        data[name+'_to_idx'][word] = key
    return data

def getDataset(params):
    data={}
    vocab_word, vocab_char = getVocab(params)
    data['vocab_word'] = vocab_word[:params['max_features']]
    data['vocab_char'] = vocab_char
    data = eval('read_{}'.format(params['dataset']))(data)
    data['x'] = data['train_x'] + data['dev_x'] + data['test_x']
    data['y'] = data['train_y'] + data['test_y'] + data['dev_y']
    classes=list(set(data['y']))
    if('NULL' in classes):
        classes.remove('NULL')
    data['classes']=classes
    print('classes',data['classes'])
    print('label distribution')
    print('label, dataset, train, dev')
    for label in data['classes']:
        print(label,data['y'].count(label),data['train_y'].count(label),data['dev_y'].count(label))
    element2idx(data, 'word')
    element2idx(data, 'char')
    DataAnalysis(data,params)
    params=load_wc(data, params)
    data=Sen2Index(data,params)
    return data,params

def load_wc(data,params):
    if(params['type']=='rand'):
        params['wv_maritx'] = []
        print('no pre-trained word2vecs')
        return params
    #path = 'models/' + params['dataset'] + '_'+params['wv']+'.pkl'
    path=''
    if(os.path.exists(path)):
        print('我是不耐烦的分界线')
        wc_matrix=pickle.load(open(path,'rb'))
    else:
        wc_matrix = []
        if(params['wv']=='word2vec'):
            word2vec = KeyedVectors.load_word2vec_format('models/GoogleNews-vectors-negative300.bin',binary=True)
        elif(params['wv']=='trained'):
            #word2vec_path='models/' + params['dataset'] + '_'+'train_word2vecs.bin'
            word2vec_path=''
            if(os.path.exists(word2vec_path)):
                word2vec=KeyedVectors.load_word2vec_format(word2vec_path,binary=True)
            else:
                model = Word2Vec(data['x'], size=params['dimension'], window=5, min_count=1, workers=4)
                word2vec = model.wv
                #word2vec.save_word2vec_format(word2vec_path, binary=True)
                print('word2vec model saving successfully!')
        for word in data['vocab_word']:
            if (word in word2vec.vocab):
                wc_matrix.append(word2vec.word_vec(word).astype('float32'))
            else:
                wc_matrix.append(np.random.uniform(-0.25, 0.25, params['dimension']).astype('float32'))
        # for unk and zero-padding
        wc_matrix.append(np.random.uniform(-0.25, 0.25, params['dimension']).astype('float32'))
        wc_matrix.append(np.zeros(params['dimension']).astype('float32'))
        #wc_matrix.append(np.random.uniform(-0.25, 0.25, 300).astype('float32'))
        print('len(word_matrix):',len(wc_matrix))
        wc_matrix=np.array(wc_matrix)
        #pickle.dump(wc_matrix,open(path,'wb'))
    params['wv_maritx']=wc_matrix
    return params

def save_models(model,params):
    path='models/{}_{}_{}_{}.pkl'.format(params['dataset'],params['model'],params['level'],params['time'])
    pickle.dump(model,open(path,'wb'))
    print('successful saved models !')

def getRMSE(prediction,true):
    rmse=0
    assert (len(prediction)==len(true))
    for i in range(len(prediction)):
        rmse+=math.pow(prediction[i]-true[i],2)
    rmse=math.sqrt(rmse/len(prediction))
    rmse=rmse/(1+rmse)
    return rmse

def getACC(prediction,true):
    acc=0
    print('prediction',prediction)
    print('true',true)
    for i in range(len(prediction)):
        if(prediction[i]==true[i]):
            acc+=1
    acc=acc/len(prediction)
    return acc

def load_models(params):
    path = 'models/{}_{}_{}_{}.pkl'.format(params['dataset'], params['model'], params['level'], params['time'])
    print('model path',path)
    if(os.path.exists(path)):
        try:
            model=pickle.load(open(path,'rb'))
            print('loaded model successfully!')
            return model
        except:
            print('error')
    else:
        print('no model finded!')
