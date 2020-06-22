# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 10:51:48 2020

@author: 12894
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 10:52:45 2020

@author: 12894
"""

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import *
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_accuracy
from keras.preprocessing.sequence import pad_sequences
from keras_bert import load_trained_model_from_checkpoint
from keras_bert import Tokenizer
from Model import DealWithData
import keras.backend as K
import tensorflow as tf


label = {}
_label = {}
max_seq_length = 160
lstmDim = 64

config_path = './chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = './chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = './chinese_L-12_H-768_A-12/vocab.txt'
label_path = './Parameter/tag_dict.txt'

f_label = open(label_path, 'r+', encoding='utf-8')
for line in f_label:
    content = line.strip().split()
    label[content[0].strip()] = content[1].strip()
    _label[content[1].strip()] = content[0].strip()
# dict
vocab = {}
with open(dict_path, 'r+', encoding='utf-8') as f_vocab:
    for line in f_vocab.readlines():
        vocab[line.strip()] = len(vocab)


# 预处理输入数据
def PreProcessInputData(text):
    tokenizer = Tokenizer(vocab)
    word_labels = []
    seq_types = []
    for sequence in text:
        code = tokenizer.encode(first=sequence, max_len=max_seq_length)
        word_labels.append(code[0])
        seq_types.append(code[1])
    return word_labels, seq_types


# 预处理结果数据
def PreProcessOutputData(text):
    tags = []
    for line in text:
        tag = [0]
        for item in line:
            tag.append(int(label[item.strip()]))
        tag.append(0)
        tags.append(tag)

    pad_tags = pad_sequences(tags, maxlen=max_seq_length, padding="post", truncating="post")
    result_tags = np.expand_dims(pad_tags, 2)
    return result_tags


def load_model(path):
    bert = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=max_seq_length)
    x1 = Input(shape=(None,))
    x2 = Input(shape=(None,))
    bert_out = bert([x1, x2])
    lstm_out = Bidirectional(LSTM(lstmDim,
                                  return_sequences=True,
                                  dropout=0.2,
                                  recurrent_dropout=0.2))(bert_out)
    crf_out = CRF(len(label), sparse_target=True)(lstm_out)
    model = Model([x1, x2], crf_out)
    model.load_weights(path)
    return model

#best_model1
model = load_model('best_causality_ner_model.h5')


def Id2Label(ids):
    result = []
    for id in ids:
        result.append(_label[str(id)])
    return result


def Vector2Id(tags):
    result = []
    for tag in tags:
        result.append(np.argmax(tag))
    return result


def extract_items(sentence):
    sentence = sentence[:max_seq_length - 1]
    labels, types = PreProcessInputData([sentence])
    tags = model.predict([labels, types])[0]
    result = []
    for i in range(1, len(sentence) + 1):
        result.append(tags[i])
    result = Vector2Id(result)
    tag = Id2Label(result)
    return tag
