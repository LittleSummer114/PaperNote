#encoding:utf-8
import argparse
import preprocess
import model as m
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from torch.autograd import Variable
import time
import os
import math
import pandas as pd
import numpy as np
np.random.seed(7)
from model import *

def train(data,params):
    #move model to cuda should before optimizer
    if (params['gpu'] == -1):
        model=getattr(m,params['model'])(**params)
    else:
        model=getattr(m,params['model'])(**params).cuda(params['gpu'])
    parameters=filter(lambda p:p.requires_grad,model.parameters())
    optimizer=getattr(optim,params['optimizer'])(parameters,params['learning_rate'])
    criterian=nn.CrossEntropyLoss()
    epochs=[]
    dev_acc_epochs = []
    train_metric_epochs = []
    train_loss_epochs = []
    dev_loss_epochs = []
    for e in range(params['epoch']):
        start=time.time()
        count=0
        train_loss=0
        for i in range(0,len(data['train_x_word']),params['batch_size']):
            count+=1
            batch_range = min(params['batch_size'],len(data['train_x_word'])-i)
            batch_x_word = data['train_x_word'][i:i+batch_range]
            batch_x_char = data['train_x_char'][i:i+batch_range]
            batch_y = data['train_y_word'][i:i+batch_range]
            if(params['gpu'] == -1):
                batch_x_word = Variable(torch.LongTensor(batch_x_word))
                batch_x_char = Variable(torch.LongTensor(batch_x_char))
                batch_y = Variable(torch.LongTensor(batch_y))
            else:
                batch_x_word = Variable(torch.LongTensor(batch_x_word)).cuda(params['gpu'])
                batch_x_char = Variable(torch.LongTensor(batch_x_char)).cuda(params['gpu'])
                batch_y = Variable(torch.LongTensor(batch_y)).cuda(params['gpu'])
            optimizer.zero_grad()
            model.train()
            if(params['model']=='CNN2'):
                pred=model(batch_x_word,batch_x_char)
            else:
                if(params['level']=='word'):
                    pred=model(batch_x_word)
                elif(params['level']=='char'):
                    pred=model(batch_x_char)
            loss=criterian(pred,batch_y)
            temp_y,temp_pred=batch_y.cpu().data.numpy(),pred.cpu().data.numpy()
            train_metric=getattr(preprocess,'get{}'.format(params['metric']))(np.argmax(temp_pred,axis=1),temp_y)
            train_loss+=loss.data[0]
            loss.backward()
            optimizer.step()
        train_loss=train_loss/count
        if(len(data['test_x_word'])>params['threshold']):
            dev_acc,dev_loss,dev_Rest=test_Batch(data,model,params,mode='dev')
        else:
            dev_acc,dev_loss,dev_Rest=test(data,model,params,mode='dev')
        dev_acc_epochs.append(dev_acc)
        train_metric_epochs.append(train_metric)
        train_loss_epochs.append(train_loss)
        dev_loss_epochs.append(dev_loss)
        epochs.append(e)
        end = time.time()
        cost_time = str(end - start)
        if(e==0):
            print('TIME FOR EACH EPOCH', cost_time)
        print('EPOCH:',e,'/ train_loss: ',train_loss,'/ train_metric: ',train_metric,'/ dev_'+params['metric']+': ',dev_acc,'/ dev_loss: ',dev_loss)
    result_params = params.copy()
    temp = time.strftime('%Y-%m-%d %H:%M', time.localtime(time.time()))
    result_params['time'] = temp
    temp_result = 'data/' + params['dataset'] + '/' + params['model'] + '_' + params['mode'] + '_'+ temp +'_DevResults.csv'
    test_result = 'data/' + params['dataset'] + '/' + params['model'] + '_' + params[
        'mode'] + '_' + temp + '_TestResults.csv'
    if (len(data['test_x_word']) > params['threshold']):
        test_acc, test_loss, test_Rest = test_Batch(data, model, params, mode='test')
    else:
        test_acc, test_loss, test_Rest = test(data, model, params, mode='test')
    dev_Rest.to_csv(temp_result,index=False,encoding='utf-8')
    test_Rest.to_csv(test_result, index=False, encoding='utf-8',header=False)
    if(params['purpose']=='Experiment'):
        params['test_acc']=test_acc
    params['time']=temp
    picturepath='data/'+params['dataset']+'/'+temp+'.jpg'
    plt.clf()
    plt.xlabel('Epoch')
    plt.ylabel('Evalution Metric')
    plt.plot(epochs,train_loss_epochs,'r-',lw=2,label='train_loss')
    plt.plot(epochs, train_metric_epochs, 'b-', lw=2, label='train_' + params['metric'])
    plt.plot(epochs, dev_acc_epochs, 'y-', lw=2,label='dev_'+params['metric'])
    plt.plot(epochs, dev_loss_epochs, 'g-', lw=2,label='dev_loss')
    plt.legend()
    plt.savefig(picturepath)
    result_params.pop('wv_maritx')
    result_params['dev_acc']=dev_acc
    columns=[]
    content=[]
    for param in result_params:
        columns.append(param)
        content.append(result_params[param])
    result_path='result/result_'+result_params['dataset']+'.csv'
    if(os.path.exists(result_path)):
        result=pd.read_csv(result_path,encoding='utf-8')
    else:
        result=pd.DataFrame(columns=columns)
    result=result.append(pd.DataFrame([content],columns=columns))
    result.to_csv(result_path,index=False,encoding='utf-8')
    return model,dev_acc

def test_Batch(data,model,params,mode='dev'):
    dev_Rest = pd.DataFrame()
    model.eval()
    loss=0
    count=0
    metric = 0
    pred = []
    true = []
    criterian=nn.CrossEntropyLoss()
    for i in range(0,len(data[mode+'_x']),params['batch_size']):
        count+=1
        batch_range=min(params['batch_size'],len(data[mode+'_x'])-i)
        x_word = data[mode + '_x_word'][i: i + batch_range]
        x_char = data[mode + '_x_char'][i: i + batch_range]
        y = data[mode + '_y_word'][i:i+batch_range]
        if (params['gpu'] == -1):
            x_word, x_char, y = Variable(torch.LongTensor(x_word)), Variable(torch.LongTensor(x_char),Variable(torch.LongTensor(y)))
        else:
            x_word = Variable(torch.LongTensor(x_word)).cuda(params['gpu'])
            x_char = Variable(torch.LongTensor(x_char)).cuda(params['gpu'])
            y = Variable(torch.LongTensor(y).cuda(params['gpu']))
        if (params['model'] == 'CNN2'):
            batch_pred = model(x_word, x_char)
        else:
            if (params['level'] == 'word'):
                batch_pred = model(x_word)
            elif (params['level'] == 'char'):
                batch_pred = model(x_char)
        if(params['purpose']=='Experiment' or mode=='dev'):
            temp = criterian(batch_pred, y)
            loss += temp.data[0]
            y = y.cpu().data.numpy()
            true = np.concatenate((true, y), axis=0)
        batch_pred = batch_pred.cpu().data.numpy()
        batch_pred = np.argmax(batch_pred, axis=1)
        pred = np.concatenate((batch_pred,pred),axis=0)
    if(data.get(mode+'_Id') is not None):
       dev_Rest['Id']=data[mode+'_Id']
    if(params['purpose']=='Experiment'or mode=='dev'):
        dev_Rest['Discuss'] = data[mode + '_Discuss']
        dev_Rest['true'] = true
        metric = getattr(preprocess, 'get{}'.format(params['metric']))(pred, true)
    dev_Rest['pre'] = pred
    loss=loss/count
    return metric,loss,dev_Rest

def test(data,model,params,mode='dev'):
    dev_Rest = pd.DataFrame()
    model.eval()
    x_word = data[mode+'_x_word']
    x_char = data[mode + '_x_char']
    y = data[mode+'_y_word']
    if (params['gpu'] == -1):
        x_word, x_char ,y = Variable(torch.LongTensor(x_word)),Variable(torch.LongTensor(x_char)),Variable(torch.LongTensor(y))
    else:
        x_word ,x_char, y = Variable(torch.LongTensor(x_word)).cuda(params['gpu']),Variable(torch.LongTensor(x_char)).cuda(params['gpu']),Variable(torch.LongTensor(y).cuda(params['gpu']))
    if (params['model'] == 'CNN2'):
        pred = model(x_word, x_char)
    else:
        if (params['level'] == 'word'):
            pred = model(x_word)
        elif (params['level'] == 'char'):
            pred = model(x_char)
    loss=nn.CrossEntropyLoss()(pred,y).data[0]
    pred ,y =pred.cpu().data.numpy() ,y.cpu().data.numpy()
    pred=np.argmax(pred,axis=1)
    if(data.get(mode+'_Id')is not None):
       dev_Rest['Id']=data[mode+'_Id']
    if (params['purpose'] == 'Experiment' or mode=='dev'):
        dev_Rest['Discuss'] = data[mode + '_Discuss']
        dev_Rest['true'] = y
        metric = getattr(preprocess, 'get{}'.format(params['metric']))(pred, y)
    dev_Rest['true']=y
    return metric,loss,dev_Rest

def main():
    parser=argparse.ArgumentParser(description='----------TextClassification----------')
    parser.add_argument('--model',default='CNN',help='models to perform text classification, alternative:CNN_word/CNNCCA/BiLSTM/CNN2/CNN_char')
    parser.add_argument('--mode',default='train',help='mode, alternative:train/test/dev')
    parser.add_argument('--dataset',default='Travel',help='datasets, alternative:TREC/MR/Travel')
    parser.add_argument('--type', default='non-static', help='embedding type, alternative:non-static/static/non-static')
    parser.add_argument('--wv',default='trained',help='vector,trained/cv/word2vec/Glove')
    parser.add_argument('--purpose', default='Competition', help='Experiment/Competition')
    parser.add_argument('--level', default='word', help='word/char')
    parser.add_argument('--max_features', default=11000, help='word/char')
    parser.add_argument('--number_of_cv', default=10,type=int, help='the number of splits in cross-validation')
    parser.add_argument('--cv', default=False,action='store_true', help='whether cross-validation or not')
    parser.add_argument('--epoch',default=60,type=int,help='hyper parameter')
    parser.add_argument('--batch_size',default=50,type=int,help='hyper parameter')
    parser.add_argument('--learning_rate',default=1,type=float,help='hyper parameter')
    parser.add_argument('--metric', default='RMSE', help='evalution metrix,RMSE/ACC')
    parser.add_argument('--threshold', default=5000,type=int, help='to prevent OOM,the number of test set')
    parser.add_argument('--dimension',default=300,type=int,help='word embedding dimension')
    parser.add_argument('--dropout', default=0.5, type=float, help='the dropout rate')
    parser.add_argument('--max_sent_word_length', default=30, type=int, help='max sentence length')
    parser.add_argument('-max_sent_char_length',default=50,type=int,help='max sentence char length')
    parser.add_argument('--filter_size',default=[3,4,5],type=int,nargs='+',help='hyper parameter')
    parser.add_argument('--number_of_filters',default=[100,100,100],type=int,nargs='+',help='hyper parameter')
    parser.add_argument('--saved_models', default=True, action='store_true',help='hyper parameter')
    parser.add_argument('--early_stopping', default=False, action='store_true',help='hyper parameter')
    parser.add_argument('--gpu', default=1,type=int, help='hyper parameter')
    parser.add_argument('--optimizer', default='Adadelta', help='optimizer method,obtain from module optim,')
    parser.add_argument('--error_analysis', default=False, action='store_true',help='whether to output analysis of results ')
    options=parser.parse_args()
    params={
        'optimizer':options.optimizer,
        'model':options.model,
        'cv': options.cv,
        'level':options.level,
        'max_features': options.max_features,
        'purpose': options.purpose,
        'number_of_cv': options.number_of_cv,
        'metric': options.metric,
        'error_analysis': options.error_analysis,
        'mode': options.mode,
        'dataset': options.dataset,
        'wv': options.wv,
        'epoch':options.epoch,
        'batch_size': options.batch_size,
        'learning_rate': options.learning_rate,
        'dimension': options.dimension,
        'optimizer': options.optimizer,
        'filter_size': options.filter_size,
        'dropout': options.dropout,
        'number_of_filters': options.number_of_filters,
        'saved_models': options.saved_models,
        'early_stopping': options.early_stopping,
        'gpu': options.gpu,
        'max_sent_word_length': options.max_sent_word_length,
        'max_sent_char_length': options.max_sent_char_length,
        'type': options.type,
        'threshold':options.threshold
    }
    data,params = preprocess.getDataset(params)
    print('='*20+'INFORMATION'+'='*20)
    print('MODEL: ',options.model)
    print('MODE: ', options.mode)
    print('DATSET: ', options.dataset)
    print('WV: ', options.wv)
    print('EPOCH: ', options.epoch)
    print('BATCH_SIZE: ', options.batch_size)
    print('LEARNING_RATE: ', options.learning_rate)
    print('DIMENSION: ', options.dimension)
    print('OPTIMIZER: ', options.optimizer)
    print('FILTER_SIZE: ', options.filter_size)
    print('NUMBER_OF_FILTERS: ', options.number_of_filters)
    print('SAVED_MODELS: ', options.saved_models)
    print('EARLY_STOPPING: ', options.early_stopping)
    print('GPU: ', options.gpu)
    print('VOCABULARY_WORD_SIZE: ', len(data['vocab_word']))
    print('VOCABULARY_CHAR_SIZE: ', len(data['vocab_char']))
    print('=' * 20 + 'INFORMATION' + '=' * 20)
    params['classes']=data['classes']
    params['VOCABULARY_WORD_SIZE'] = len(data['vocab_word'])
    params['VOCABULARY_CHAR_SIZE'] = len(data['vocab_char'])
    if(params['cv']==True):
        print('Using Cross Validation')
        kf=StratifiedKFold(n_splits=params['number_of_cv'],shuffle=True)
        metric = 0
        for train_index,test_index in kf.split(data['x_word'],data['y_word']):
            data['train_x_word'],data['dev_x_word']=np.array(data['x_word'])[train_index].tolist(),np.array(data['x_word'])[test_index].tolist()
            data['train_y_word'],data['dev_y_word']=np.array(data['y_word'])[train_index].tolist(),np.array(data['y_word'])[test_index].tolist()
            data['train_x_char'], data['dev_x_char'] = np.array(data['x_char'])[train_index].tolist(), np.array(data['x_char'])[test_index].tolist()
            data['train_y_char'], data['dev_y_char'] = np.array(data['y_char'])[train_index].tolist(), np.array(data['y_char'])[test_index].tolist()
            model,dev_acc=train(data,params)
            metric+=dev_acc
        metric=metric/params['number_of_cv']
        print('Dataset: ',params['dataset'],'final result: ', metric)
    else:
        if (params['mode'] == 'train'):
            model, dev_acc = train(data, params)
            if (params['saved_models'] == True):
                preprocess.save_models(model, params)
        else:
            model = preprocess.load_models(params)
            if (len(data['test_x_word']) < params['threshold']):
                metric,loss,result = eval(params['mode'])(data, model, params, mode='test')
                print(params['dataset']+params['model']+ 'metric:', metric)
            else:
                print('BATCH TEST...')
                metric, loss, result = eval(params['mode'] + '_Batch')(data, model, params, mode='test')
                print(params['dataset'] + params['model'] + 'metric:', metric)

if __name__=='__main__':
    main()
