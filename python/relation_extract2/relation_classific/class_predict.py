# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 14:28:55 2020

@author: 12894
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 09:36:35 2020

@author: 12894
"""
import pandas as pd
import numpy as np
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from keras.metrics import top_k_categorical_accuracy
from keras.layers import *
from keras.callbacks import *
from keras.models import Model
import keras.backend as K
import tensorflow as tf

# 预训练好的模型
base_path = r"resc/chinese_L-12_H-768_A-12"
config_path = f"{base_path}/bert_config.json"
checkpoint_path = f"{base_path}/bert_model.ckpt"
dict_path = f"{base_path}/vocab.txt"
nclass = 3

g0 = tf.Graph()
ss0 = tf.Session(graph=g0)

# 将词表中的词编号转换为字典
token_dict = {}
with open(dict_path, 'r+', encoding='utf-8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)


# 重写tokenizer
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


# bert模型设置

def load_model(path):
    with ss0.as_default():
        with ss0.graph.as_default():
            bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path,
                                                            seq_len=None)  # 加载预训练模型
            for l in bert_model.layers:
                l.trainable = True

            x1_in = Input(shape=(None,))
            x2_in = Input(shape=(None,))

            x = bert_model([x1_in, x2_in])
            x = Lambda(lambda x: x[:, 0])(x)  # 取出[CLS]对应的向量用来做分类
            p = Dense(nclass, activation='softmax')(x)
            model = Model([x1_in, x2_in], p)
            model.load_weights(path)
            return model


def class_pre(model, test1, test2):
    t1, t1_ = tokenizer.encode(first=test1, second=test2)
    T1, T1_ = np.array([t1]), np.array([t1_])
    with ss0.as_default():
        with ss0.graph.as_default():
            _prob = model.predict([T1, T1_])
            prob = np.argmax(_prob)
    return prob
