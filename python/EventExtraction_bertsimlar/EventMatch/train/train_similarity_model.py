#! -*- coding:utf-8 -*-
import os
import sys

file_path = os.path.abspath(__file__)
dir_path = os.path.dirname(file_path)
sys.path.append(dir_path)

import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from resources.bert4keras_v01.utils import ArgConfig, data_generator, print_arguments, init_log, load_data
from resources.bert4keras_v01.backend import keras, set_gelu, K
from resources.bert4keras_v01.tokenizer import Tokenizer
from resources.bert4keras_v01.bert import build_bert_model
from resources.bert4keras_v01.optimizers import Adam
from resources.bert4keras_v01.snippets import sequence_padding, get_all_attributes
from keras.layers import Dense, Dropout

import logger

# 更新模型的所有层，主要是第三方包与keras的对接处理
locals().update(get_all_attributes(keras.layers))
from keras.models import save_model, load_model

LOG = logger.Logger("info")

def run(args: ArgConfig):
    '''
    传入模型参数，进行模型训练，保存模型文件为.h5，评估模型精确率、召回率、F1值
    :param args: 模型训练参数
    :return: None
    '''
    # 加载模型参数
    set_gelu(args.gelu)
    # 分字最大长度
    maxlen = args.max_length
    # 批量大小
    batch_size = args.batch_size
    # 模型参数路径
    config_path = args.config_path
    # 初始化模型路径
    checkpoint_path = args.init_checkpoint
    # 模型子带你路径
    dict_path = args.vocab_path
    # 训练后模型保存路径
    trained_model_path = args.trained_model_path
    # 加载数据
    train_data = load_data(args.train_data_dir)
    valid_data = load_data(args.valid_data_dir)
    test_data = load_data(args.test_data_dir)

    # 建立分词
    tokenizer = Tokenizer(dict_path)

    # 加载预训练模
    bert = build_bert_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        with_pool=True,
        return_keras_model=False,
    )

    output = Dropout(rate=0.1)(bert.model.output)
    output = Dense(units=2,
                   activation='softmax',
                   kernel_initializer=bert.initializer)(output)

    model = keras.models.Model(bert.model.input, output)
    model.summary()

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(2e-5),  # 用足够小的学习率
        metrics=['accuracy'],
    )

    # 转换数据
    train_generator = data_generator(train_data, tokenizer, max_length=maxlen, batch_size=batch_size)
    valid_generator = data_generator(valid_data, tokenizer, max_length=maxlen, batch_size=batch_size)
    test_generator = data_generator(test_data, tokenizer, max_length=maxlen, batch_size=batch_size)

    def evaluate(data):
        '''
        对传入的数据进行预测，并评估模型效果
        :param data: 传入测试数据[batch_token_ids, batch_segment_ids], batch_labels
        :return: 评估指标 精确率 找回率 f1
        '''
        y_pred_total = []
        y_true_total = []
        for x_true, y_true in data:
            y_pred = model.predict(x_true).argmax(axis=1)
            y_true = y_true[:, 0]
            y_pred_total += np.reshape(y_pred, (-1,)).tolist()
            y_true_total += np.reshape(y_true, (-1,)).tolist()
        return precision_recall_fscore_support(y_true_total, y_pred_total, beta=1.0, average='macro')

    class Evaluator(keras.callbacks.Callback):
        '''
        继承底层的callback类，构建模型选择器
        '''
        def __init__(self):
            '''
            初始化最优的模型指标为0
            '''
            self.best_val_acc = 0.

        def on_epoch_end(self, epoch, logs=None):
            '''
            每个循环结束时进行处理，选择f1指标最大的模型进行保存
            :param epoch: 循环数
            :param logs: 日志
            :return: None
            '''
            val_precision, val_recall, val_f_score, _ = evaluate(valid_generator)
            if val_f_score > self.best_val_acc:
                self.best_val_acc = val_f_score
                save_model(model, trained_model_path, include_optimizer=True)
            test_precision, test_recall, test_f_score, _ = evaluate(test_generator)
            LOG.info(u'val_precision: %05f, val_recall: %05f, val_f_score: %05f, best_val_f_score: %05f, test_precision: %05f, test_recall: %05f, test_f_score: %05f\n'
                  % (val_precision, val_recall, val_f_score, self.best_val_acc, test_precision, test_recall, test_f_score))
    # 构建评估对象
    evaluator = Evaluator()
    # 开始模型训练
    model.fit_generator(train_generator.forfit(),
                        steps_per_epoch=len(train_generator),
                        epochs=args.epoch,
                        verbose=1,
                        callbacks=[evaluator])
    # 重新加载模型参数
    model = load_model(trained_model_path, compile=False)
    # 使用测试集评估模型效果
    precision, recall, f_score, _ = evaluate(test_generator)
    LOG.info(u'final test precision: %05f, recall: %05f, f_score: %05f\n' % (precision, recall, f_score))

if __name__ == '__main__':
    args = ArgConfig()
    args = args.build_conf()

    print_arguments(args)
    run(args)
