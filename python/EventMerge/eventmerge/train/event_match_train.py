#!/usr/bin/env python
# coding:utf-8
# 事件归并模型训练
import os
import numpy as np
import json
from keras.models import Model
from keras.layers import Dense, Concatenate, Input, Dropout
from keras.callbacks import Callback
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.backend import K
from keras.optimizers import Adam
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from hrconfig import merge_train_config
from EventMerge.train.utils.data_utils import load_data, generate_vec, load_vec_data, DataGenerator
from feedwork.utils import logger as LOG

# 参数类
CONFIG = merge_train_config.Config()

# 分字器
TOKENIZER = Tokenizer(CONFIG.vocab, do_lower_case=True)
# bert模型主体
BERT_MODEL = build_transformer_model(CONFIG.bert_config, CONFIG.bert_model_path,
                                     return_keras_model=False)  # 建立bert模型，加载权重

# 设置bert底层的参数不可训练
for l in BERT_MODEL.layers:
    BERT_MODEL.model.get_layer(l).trainable = False

# 加载数据
LOG.info("开始加载训练集数据。。。")
TRAIN_DATA = load_data(CONFIG.train_data_path)
LOG.info("开始加载测试集数据。。。")
DEV_DATA = load_data(CONFIG.dev_data_path)
LOG.info("原始数据加载完成！")

LOG.info("开始转化训练集数据。。。")
generate_vec("train", TRAIN_DATA, TOKENIZER, BERT_MODEL, CONFIG.vector_data_path, CONFIG.vector_id_dict_dir)
LOG.info("开始转化测试集数据。。。")
generate_vec("dev", DEV_DATA, TOKENIZER, BERT_MODEL, CONFIG.vector_data_path, CONFIG.vector_id_dict_dir)
LOG.info("数据转换完成！")

LOG.info("开始加载向量数据。。。")
VECTOR_DATA = load_vec_data(CONFIG.vector_data_path)
LOG.info("向量数据加载完成！")

# 创建数据生成器
TRAIN_D = DataGenerator("train", TRAIN_DATA, VECTOR_DATA, CONFIG.vector_id_dict_dir, CONFIG.batch_size, shuffle=True)


def build_model():
    """
    构建相似度匹配模型主体，返回模型对象。
    :return: match_mdoel 模型对象
    """
    # 搭建模型主体
    x1_in = Input(shape=(768,))
    x2_in = Input(shape=(768,))

    # 拼接融合
    t = Concatenate(axis=1)([x1_in, x2_in])
    t = Dense(768, activation='relu')(t)
    t = Dropout(0.4)(t)
    # 计算两个句子的相似度
    output = Dense(2, activation='softmax')(t)
    # 相似度模型
    match_model = Model([x1_in, x2_in], [output])
    match_model.compile(loss="sparse_categorical_crossentropy", optimizer=Adam(CONFIG.learning_rate),
                        metrics=["accuracy"])
    match_model.summary()

    return match_model


# 构建相似度模型
MATCH_MODEL = build_model()


def measure():
    """
    调用相似度模型，对测试集数据进行相似度计算，返回测试集相似度计算结果。
    :return: test_model_pred(lsit)测试集相似度列表
    """
    # 测试集模式
    mode = "dev"
    # 测试集数据保存字典
    dict_name = f"{mode}_dict.json"
    dict_path = os.path.join(CONFIG.vector_id_dict_dir, dict_name)

    # 加载字典
    with open(dict_path, "r", encoding="utf-8") as f:
        dev_dict = f.read()
        dev_dict = json.loads(dev_dict)

    test_model_pred = []
    for i in range(len(DEV_DATA)):
        x1 = DEV_DATA[i].split("\t")[0]
        x2 = DEV_DATA[i].split("\t")[1]

        key = str(i // 10000)

        x1_id = dev_dict[key].index(x1)
        x2_id = dev_dict[key].index(x2)

        # 验证集向量保存位置
        file_name = "{}_{}.npy".format(mode, key)
        x1 = VECTOR_DATA[file_name][x1_id]
        x2 = VECTOR_DATA[file_name][x2_id]

        x = MATCH_MODEL.predict([np.array([x1]), np.array([x2])])
        test_model_pred.append(np.argmax(x))

    return test_model_pred


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
        第一个epoch用来warmup，第二个epoch把学习率降到最低
        :param batch: 批次大小
        :param logs:日志信息
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
        每个训练批次结束时运行，如果得到的准确率是最大的则进行模型保存
        :param epoch: 循环次数
        :param logs: 日志信息
        :return: None
        """
        precision, recall, f_score, accuracy = self.evaluate()
        self.F1.append(f_score)
        if f_score > self.best:
            self.best = f_score
            MATCH_MODEL.save(CONFIG.trained_model_path, include_optimizer=True)
        LOG.info(
            f'precision: {precision}.4f, recall: {recall}.4f, f_score:{f_score}.4f, accuracy:{accuracy}.4f, best f_score: {self.best}.4f\n')

    @staticmethod
    def flat_lists(lists):
        """
        对出入的列表文件进行整理，拉平
        :param lists: 列表
        :return: 拉平后的列表
        """
        all_elements = []
        for lt in lists:
            all_elements.extend(lt)
        return all_elements

    @staticmethod
    def evaluate():
        """
        评估函数
        :return:准确率、精确率、召回率、f值
        """
        y_pred = measure()

        y_true = [int(once.split("\t")[2]) for once in DEV_DATA]
        y_true = np.array(y_true).reshape((-1,)).tolist()
        y_pred = np.array(y_pred).reshape((-1,)).tolist()

        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f_score, true_sum = precision_recall_fscore_support(y_true, y_pred, beta=1.0,
                                                                               average='macro')

        return precision, recall, f_score, accuracy


def model_train():
    """
    进行模型训练
    :return: None
    """
    # 构建模型评估对象
    evaluator = Evaluate()
    # 模型训练
    MATCH_MODEL.fit_generator(TRAIN_D.__iter__(), steps_per_epoch=TRAIN_D.__len__(), epochs=100, callbacks=[evaluator])
    # 模型参数重载
    MATCH_MODEL.load_weights(CONFIG.trained_model_path)
    # 调用评估方法，测试模型效果
    precision, recall, f_score, accuracy = evaluator.evaluate()
    LOG.info(f'precision: {precision}.4f, recall: {recall}.4f, f_score:{f_score}.4f, accuracy:{accuracy}.4f')


if __name__ == '__main__':
    model_train()
