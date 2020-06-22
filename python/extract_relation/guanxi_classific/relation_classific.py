# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 09:36:35 2020

@author: 12894
"""
import pandas as pd
import numpy as np
#import codecs
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from keras.metrics import top_k_categorical_accuracy
from keras.layers import *
from keras.callbacks import *
from keras.models import Model
import keras.backend as K
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.metrics import precision_score, recall_score, f1_score
import random

causality = []
with open('./guanlianci_ner/data/causality/samples_causality_add.txt', encoding='utf-8') as f:
    fs = list(set(f.readlines()))
    for line in fs:
        if len(line.strip().split('\t')) == 6:
            causality.append((line.strip().split('\t')[4].replace('|',''), line.strip().split('\t')[5].replace('|',''), to_categorical(1, 3)))
            causality.append((line.strip().split('\t')[5].replace('|',''), line.strip().split('\t')[4].replace('|',''), to_categorical(2, 3)))

with open('./samples_causality_pos_svos.txt', encoding='utf-8') as f:
    fs = list(set(f.readlines()))
    for line in fs:
        if len(line.strip().split('\t')) == 5:
            causality.append((line.strip().split('\t')[3].replace('|',''), line.strip().split('\t')[4].replace('|',''), to_categorical(1, 3)))
            causality.append((line.strip().split('\t')[4].replace('|',''), line.strip().split('\t')[3].replace('|',''), to_categorical(2, 3)))

          
causality_neg = []
with open('./samples_causality_svos_neg.txt', encoding='utf-8') as f:
    fs = list(set(f.readlines()))
    for line in fs:
        if len(line.strip().split()) == 3:
            causality_neg.append((line.strip().split()[1].replace('|',''), line.strip().split()[2].replace('|',''), to_categorical(0, 3)))

with open('./guanlianci_ner/data/causality/samples_causality_neg_add.txt', encoding='utf-8') as f:
    fs = list(set(f.readlines()))
    for line in fs:
        if len(line.strip().split()) == 3:
            causality_neg.append((line.strip().split()[1].replace('|',''), line.strip().split()[2].replace('|',''), to_categorical(0, 3)))

causas_train = causality + random.sample(causality_neg, len(causality)//2)

random_order = list(range(len(causas_train)))
np.random.shuffle(random_order)
train_line = [causas_train[j] for i, j in enumerate(random_order) if i % 5 != 0]
test_line = [causas_train[j] for i, j in enumerate(random_order) if i % 5 == 0]


maxlen = 60  #设置序列长度为60，要保证序列长度不超过512
epoch = 2
nclass = 3
#预训练好的模型
config_path = "D:/myproject/chinese_L-12_H-768_A-12/bert_config.json"
checkpoint_path =  "D:/myproject/chinese_L-12_H-768_A-12/bert_model.ckpt"
dict_path =  "D:/myproject/chinese_L-12_H-768_A-12/vocab.txt"

 

#将词表中的词编号转换为字典
token_dict = {}
with open(dict_path, 'r+', encoding='utf-8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)


#重写tokenizer        
class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')  # 用[unused1]来表示空格类字符
            else:
                R.append('[UNK]')  # 不在列表的字符用[UNK]表示
        return R

tokenizer = OurTokenizer(token_dict)

 

#让每条文本的长度相同，用0填充
def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])

 

#data_generator只是一种为了节约内存的数据方式
class data_generator:
    def __init__(self, data, batch_size=10, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps


    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            if self.shuffle:
                np.random.shuffle(idxs)
            X1, X2, Y = [], [], []
            for i in idxs:
                d = self.data[i]
                text1 = d[0][:maxlen]
                text2 = d[1][:maxlen]
                x1, x2 = tokenizer.encode(first=text1, second=text2)
                y = d[2]
                X1.append(x1)
                X2.append(x2)
                Y.append([y])
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    Y = seq_padding(Y)
                    yield [X1, X2], Y[:, 0, :]
                    [X1, X2, Y] = [], [], [] 

#bert模型设置

def build_bert(nclass):
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)  #加载预训练模型
    for l in bert_model.layers:
        l.trainable = True

    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))

    x = bert_model([x1_in, x2_in])
    x = Lambda(lambda x: x[:, 0])(x) # 取出[CLS]对应的向量用来做分类
    p = Dense(nclass, activation='softmax')(x)
    model = Model([x1_in, x2_in], p)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(1e-5),    #用足够小的学习率
                  metrics=['accuracy'])
    return model

train_D = data_generator(train_line)
valid_D = data_generator(test_line)


model = build_bert(nclass)

class Evaluate(Callback):
    def __init__(self):
        self.best = 0.

    def on_epoch_end(self, epoch, logs=None):
        p, r, f1 = self.evaluate()
        if f1 > self.best:
            self.best = f1
            model.save('best_contrast_class_model.h5', include_optimizer=True)
        print('epoch: %d, p: %.4f, r: %.4f, f1: %.4f, best: %.4f\n' % (epoch, p, r, f1, self.best))

    def evaluate(self):
        true_lists = []
        probs = []
        for i in range(len(test_line)):
            test1 = test_line[i][0]
            test2 = test_line[i][1]
            true_lists.append(np.argmax(test_line[i][2]))
            t1, t1_ = tokenizer.encode(first=test1, second=test2)
            T1, T1_ = np.array([t1]), np.array([t1_])
            _prob = model.predict([T1, T1_])
            prob = np.argmax(_prob)
            probs.append(prob)

        p = precision_score(true_lists,probs,average='macro') 
        r = recall_score(true_lists,probs,average='macro') 
        f1 = f1_score(true_lists,probs,average='macro') 
        return p, r, f1

def model_test(test_line):
    """输出测试结果
    """
    with open ('test_contrast_class_pre.txt', 'w', encoding = 'utf-8') as f:
        for i in range(len(test_line)):
            test1 = test_line[i][0]
            test2 = test_line[i][1]
            t1, t1_ = tokenizer.encode(first=test1, second=test2)
            T1, T1_ = np.array([t1]), np.array([t1_])
            _prob = model.predict([T1, T1_])
            prob = np.argmax(_prob)
            f.write(test1 + '\t' + test2 + '\t' + str(np.argmax(test_line[i][2])) + '\t' + str(prob) + '\n')
            
model_test(test_line)

if __name__ == '__main__':
    evaluator = Evaluate()
    model.fit_generator(
    train_D.__iter__(),
    steps_per_epoch=len(train_D),
    epochs= epoch,
    validation_data=valid_D.__iter__(),
    validation_steps=len(valid_D),
    verbose=1,
    shuffle=True,
    callbacks=[evaluator]
    )   
    model_test(test_line)
    p, r, f1 = evaluator.evaluate()
    print("p:{}, r:{}, f1:{}".format(p, r, f1))
