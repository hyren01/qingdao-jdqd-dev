#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
from keras.layers import Lambda, Dense, Dropout
from keras.models import Model
from keras.callbacks import Callback
from sklearn.metrics import accuracy_score
from bert4keras.models import build_transformer_model
from bert4keras.backend import K
from bert4keras.optimizers import Adam
from feedwork.utils import logger as LOG
from hrconfig import event_cameo_train_config
from event_extract.train.utils.utils import get_bert_tokenizer, generate_trained_model_path
from event_extract.train.utils.event_cameo_data_util import DataGenerator, get_data

# 构建默认图和会话
GRAPH = tf.Graph()
SESS = tf.Session(graph=GRAPH)

# 模型配置类
CONFIG = event_cameo_train_config.Config()

# 创建训练后的保存路径,按照当前日期创建模型保存文件夹
TRAINED_MODEL_PATH = generate_trained_model_path(CONFIG.trained_model_dir, CONFIG.trained_model_name)
# 加载训练集和验证集
TRAIN_DATA, DEV_DATA, ID2LABEL, LABEL2ID = get_data(CONFIG.train_data_path, CONFIG.dev_data_path, CONFIG.label2id_path,
                                                    CONFIG.id2label_path)
# 加载bert分字器
TOKENIZER = get_bert_tokenizer(CONFIG.dict_path)


def build_model():
    """
    构建模型主体。
    :return: 模型对象
    """
    with SESS.as_default():
        with SESS.graph.as_default():
            # 搭建bert模型主体
            bert_model = build_transformer_model(
                config_path=CONFIG.config_path,
                checkpoint_path=CONFIG.checkpoint_path,
                return_keras_model=False,
                model=CONFIG.model_type
            )

            # l为模型内部的层名，格式为--str
            for l in bert_model.layers:
                bert_model.model.get_layer(l).trainable = True

            t = Lambda(lambda x: x[:, 0])(bert_model.model.output)  # 取出[CLS]对应的向量用来做分类
            t = Dropout(CONFIG.drop_out_rate)(t)
            cameo_out_put = Dense(len(ID2LABEL), activation='softmax')(t)

            cameo_model = Model(bert_model.model.inputs, cameo_out_put)

            cameo_model.compile(loss='sparse_categorical_crossentropy',
                                optimizer=Adam(CONFIG.learning_rate),
                                metrics=['accuracy'])

            cameo_model.summary()

    return cameo_model


# 构建模型
MODEL = build_model()


def measure():
    """
    调用模型，预测验证集结果。
    :return: result（list）验证集预测结果
    """
    result = []

    for once in DEV_DATA:
        text = once[0]
        text_token_ids, text_segment_ids = TOKENIZER.encode(first_text=text[:CONFIG.maxlen])
        pre = MODEL.predict([np.array([text_token_ids]), np.array([text_segment_ids])])
        pre = np.argmax(pre)
        result.append(pre)

    return result


class Evaluate(Callback):
    """
    继承Callback类，改写内部方法,调整训练过程中的学习率，随着训练步数增加，逐渐减小学习率，根据指标保存最优模型。

    """

    def __init__(self):
        Callback.__init__(self, )
        self.F1 = []
        self.best = 0.
        self.passed = 0
        self.stage = 0

    def on_batch_begin(self, batch, logs=None):
        """
        在每个批次开始，判断批次步数，来调整训练时的学习率
        :param batch: 批次步数
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
        每个循环结束时判断指标是否最好，是则将模型保存下来。
        :param epoch: 循环次数
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
         对传入的列表进行拉平。
        :param lists: (list)传入二维列表
        :return: all_elements（list)拉平后的列表
        """

        all_elements = []
        for lt in lists:
            all_elements.extend(lt)
        return all_elements

    @staticmethod
    def evaluate():
        """
        评估模型在验证集上的准确率。
        :return: accuracy(float)验证集准确率
        """

        y_pred = measure()
        y_true = [int(once[1]) for once in DEV_DATA]

        accuracy = accuracy_score(y_true, y_pred)

        return accuracy


def model_train():
    """
    进行模型训练。
    :return: None
    """
    with SESS.as_default():
        with SESS.graph.as_default():
            train_d = DataGenerator(TOKENIZER, CONFIG.maxlen, TRAIN_DATA, CONFIG.batch_size)
            evaluator = Evaluate()
            # 模型训练
            MODEL.fit_generator(train_d.__iter__(), steps_per_epoch=train_d.__len__(), epochs=40, callbacks=[evaluator])
            # 模型重载
            MODEL.load_weights(TRAINED_MODEL_PATH)
            accuracy = evaluator.evaluate()
            LOG.info(f"accuracy:{accuracy}")


if __name__ == '__main__':
    model_train()
