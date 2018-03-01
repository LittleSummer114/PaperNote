#encoding:utf-8
import os
'''
1. 确定 learning rate
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
wvs=[
    'trained'
]
epochs=[
    50
]
batch_sizes=[
    500
             ]
learning_rates=[
    1
]
metrics=['RMSE']
dimensions=[
    256
]
max_sent_word_lengths=[
    200
]
max_sent_char_lengths=[
    200
]
filter_sizes=[
    ['3','4','5']
]
num_of_filters=[
    ['100','100','100']
]
gpu={'Travel':3}
optimizers=['Adadelta','SGD']

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
                                                                  '--optimizer {}'
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
                                                            optimizer
                                                        ))

