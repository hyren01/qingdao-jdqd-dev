# coding: utf-8
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from utils import ArgConfig, data_generator, print_arguments, init_log, load_data
from .bert4keras_v01.tokenizer import Tokenizer
from .bert4keras_v01.backend import keras, set_gelu, K
from .bert4keras_v01.bert import build_bert_model
from .bert4keras_v01.optimizers import Adam
from .bert4keras_v01.snippets import sequence_padding, get_all_attributes
import os

locals().update(get_all_attributes(keras.layers))  # from keras.layers import *
from keras.models import load_model
from utils import data_generator
# from keras import backend as K1


maxlen = 256
batch_size = 1


vocab_path = "D:/work/ArticleExtract_bertsimlar/resources/model/vocab.txt"
model_path = "D:/work/ArticleExtract_bertsimlar/resources/model/new_best_val_acc_model.h5"

tokenizer = Tokenizer(vocab_path)
model = load_model(model_path, compile=False)


def get_result(test_data):
    '''
    传入测试数据，进行预测
    :param test_data:
    :return:
    '''
    def evaluate(data):
        results = []
        for x_true, y_true in data:
            y_pred = model.predict(x_true)
            results += np.reshape(y_pred[:, 1], (-1,)).tolist()
        return results

    test_generator = data_generator(test_data, tokenizer, max_length=maxlen, batch_size=batch_size)
    results = evaluate(test_generator)
    return results


if __name__ == '__main__':
    test_data = []
    get_result(test_data)