#!/usr/bin/env python
# -*- coding:utf-8 -*-
# 事件状态模型训练
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.layers import Input, Average, Lambda, Dense
from keras.models import Model
from keras.callbacks import Callback
from bert4keras.models import build_transformer_model
from bert4keras.backend import K
from bert4keras.layers import LayerNormalization
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from feedwork.utils import logger as LOG
from hrconfig import event_state_train_config
from event_extract.train.utils.utils import get_bert_tokenizer, seq_gather, generate_trained_model_path
from event_extract.train.utils.event_state_data_util import DataGenerator, get_data

# 构建默认图和会话
GRAPH = tf.Graph()
SESS = tf.Session(graph=GRAPH)

# 模型配置类
CONFIG = event_state_train_config.Config()

# 创建训练后的保存路径,按照当前日期创建模型保存文件夹
TRAINED_MODEL_PATH = generate_trained_model_path(CONFIG.trained_model_dir, CONFIG.trained_model_name)
# 加载训练集和验证集
DATA_LIST, DATA_PRE_TEST, TEST_DF, ID2LABEL, LABEL2ID = get_data(CONFIG.all_data_path)
# 加载bert分字器
TOKENIZER = get_bert_tokenizer(CONFIG.dict_path)


def build_model():
    """
    搭建模型主体。
    :return: 模型对象
    """
    with SESS.as_default():
        with SESS.graph.as_default():
            # 构建bert模型主体
            bert_model = build_transformer_model(
                config_path=CONFIG.config_path,
                checkpoint_path=CONFIG.checkpoint_path,
                return_keras_model=False,
                model=CONFIG.model_type
            )

            # l为模型内部的层名，格式为--str
            for l in bert_model.layers:
                bert_model.model.get_layer(l).trainable = True

            trigger_start_index = Input(shape=(1,))
            trigger_end_index = Input(shape=(1,))

            k1v = Lambda(seq_gather)([bert_model.model.output, trigger_start_index])
            k2v = Lambda(seq_gather)([bert_model.model.output, trigger_end_index])
            kv = Average()([k1v, k2v])
            t = LayerNormalization(conditional=True)([bert_model.model.output, kv])
            t = Lambda(lambda x: x[:, 0])(t)  # 取出[CLS]对应的向量用来做分类

            state_out_put = Dense(3, activation='softmax')(t)
            state_model = Model(bert_model.model.inputs + [trigger_start_index, trigger_end_index], state_out_put)

            state_model.compile(loss='categorical_crossentropy',
                                optimizer=Adam(CONFIG.learning_rate),
                                metrics=['accuracy'])

            state_model.summary()

    return state_model


# 构建模型
MODEL = build_model()
# 构建训练集以及测试集生成类
TRAIN_D = DataGenerator(TOKENIZER, CONFIG.maxlen, DATA_LIST, CONFIG.batch_size)
TEST_PRE_D = DataGenerator(TOKENIZER, CONFIG.maxlen, DATA_PRE_TEST, shuffle=False)


def extract_items():
    """
    调用训练的模型对测试集进行预测,将预测结果写入文件中
    :return: None
    """
    test_model_pred = MODEL.predict_generator(TEST_PRE_D.__iter__(), steps=len(TEST_PRE_D), verbose=1)
    test_pred = [np.argmax(x) for x in test_model_pred]
    output = pd.DataFrame(
        {'sentences': TEST_DF.sentences, 'trig': TEST_DF.trig, 'la': TEST_DF['la'], 'pre_label': test_pred})
    output.to_csv(CONFIG.pred_path, encoding='utf_8_sig', index=0)


class Evaluate(Callback):
    """
    继承Callback类，改写内部方法,调整训练过程中的学习率，随着训练步数增加，逐渐减小学习率，根据指标保存最优模型。
    """
    def __init__(self):
        Callback.__init__(self,)
        self.F1 = []
        self.best = 0.
        self.passed = 0
        self.stage = 0

    def on_batch_begin(self, batch, logs=None):
        """
        在每个批次开始，判断批次步数，来调整训练时的学习率
        :param batch: (int)批次步数
        :param logs: 日志信息
        :return: None
        """
        if self.passed < self.params['steps']:
            lr = (self.passed + 1.) / self.params['steps'] * CONFIG.learning_rate
            K.set_value(self.model.optimizer.lr, lr)
            self.passed += 1
        elif self.params['steps'] <= self.passed < self.params['steps'] * 2:
            lr = (2 - (self.passed + 1.) / self.params['steps']) * (CONFIG.learning_rate - CONFIG.min_learning_rate)
            lr += CONFIG.min_learning_rate
            K.set_value(self.model.optimizer.lr, lr)
            self.passed += 1

    def on_epoch_end(self, epoch, logs=None):
        """
         每个循环结束时判断指标是否最好，是则将模型保存下来
         :param epoch: (int)循环次数
         :param logs: 日志信息
         :return: None
        """
        accuracy = self.evaluate()
        self.F1.append(accuracy)
        if accuracy > self.best:
            self.best = accuracy
            MODEL.save(TRAINED_MODEL_PATH, include_optimizer=True)
        LOG.info(f'accuracy: {accuracy}.4f, best accuracy: {self.best}.4f\n')

    @staticmethod
    def flat_lists(lists):
        """
        对传入的列表进行拉平
        :param lists: (list)传入二维列表
        :return: 拉平后的列表
        """
        all_elements = []
        for lt in lists:
            all_elements.extend(lt)
        return all_elements

    @staticmethod
    def evaluate():
        """
        评估模型效果
        :return: 准确率
        """
        extract_items()

        pred_data = pd.read_csv(CONFIG.pred_path, engine="python", encoding="utf_8_sig")
        y_true = pred_data.loc[:, 'la']
        y_pred = pred_data.loc[:, "pre_label"]
        accuracy = accuracy_score(y_true, y_pred)

        return accuracy


def model_train():
    """
    进行模型训练。
    :return: None
    """
    with SESS.as_default():
        with SESS.graph.as_default():
            # 构造评估对象
            evaluator = Evaluate()
            # 模型训练
            MODEL.fit_generator(TRAIN_D.__iter__(), steps_per_epoch=TRAIN_D.__len__(), epochs=40, callbacks=[evaluator])
            # 模型参数重载
            MODEL.load_weights(TRAINED_MODEL_PATH)
            accuracy = evaluator.evaluate()
            LOG.info(f"accuracy:{accuracy}")


if __name__ == '__main__':
    model_train()
