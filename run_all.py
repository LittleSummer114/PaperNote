#encoding:utf-8
import os
'''
1. 确定 learning rate
2. 分别运行word char
3. 尝试看看length_feature有没有用
4. 尝试修改dimension
5. 修改filter_size
6. 修改max_features

1. 修改模型结构,变得更宽更深
'''
models=[
    'CNN'
]
modes=[
    'train'
]
datasets=[
    'Travel'
]
levels=[
   # 'word',
    'char'
]
length_features=[
    0,
    #1
]
wvs=[
    'trained'
]
epochs=[
    100
]
batch_sizes=[
    500
             ]
learning_rates=[
    0.01,
    0.03,
    0.001,
    0.003,
    0.1,
    0.3,
    1
]
metrics=['RMSE']
dimensions=[
    256
]
max_sent_word_lengths=[
    40
]
max_sent_char_lengths=[
    60
]
filter_sizes=[
    ['3','4','5']
]
num_of_filters=[
    ['100','100','100']
]
gpu={'Travel':3}
optimizers=['Adadelta']

if __name__=='__main__':
    for dataset in datasets:
        for model in models:
            for mode in modes:
                for wv in wvs:
                    for batch_size in batch_sizes:
                        for learning_rate in learning_rates:
                            for epoch in epochs:
                                for dimension in dimensions:
                                    for max_sent_word_length in max_sent_word_lengths:
                                        for filter_size in filter_sizes:
                                            for num_of_filter in num_of_filters:
                                                for optimizer in optimizers:
                                                    for max_sent_char_length in max_sent_char_lengths:
                                                        for level in levels:
                                                            for length_feature in length_features:
                                                                os.system('python3 run.py --model {} '
                                                                  '--mode {} '
                                                                  '--wv {} '
                                                                  '--batch_size {} '
                                                                  '--learning_rate {} '
                                                                  '--epoch {} '
                                                                  '--dimension {} '
                                                                  '--max_sent_word_length {} '
                                                                  '--max_sent_char_length {} '
                                                                  '--filter_size {} '
                                                                  '--number_of_filters {} '
                                                                  '--optimizer {} '
                                                                  '--level {} '
                                                                  '--length_feature {} '
                                                                  '--gpu {} '
                                                                  '--dataset {}'
                                                                  ''.format(
                                                            model,
                                                            mode,
                                                            wv,
                                                            batch_size,
                                                            learning_rate,
                                                            epoch,
                                                            dimension,
                                                            max_sent_word_length,
                                                            max_sent_char_length,
                                                            ' '.join(filter_size),
                                                            ' '.join(num_of_filter),
                                                            optimizer,
                                                            level,
                                                            length_feature,
                                                            gpu[dataset],
                                                            dataset
                                                        ))
