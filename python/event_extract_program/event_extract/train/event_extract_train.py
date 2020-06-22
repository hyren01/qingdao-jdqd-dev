#!/usr/bin/env python
# ! -*- coding:utf-8 -*-
# 事件抽取模型训练
# 使用条件批归一化进行向量融合
# 对传入的数据进行处理，不再进行字符串的查找，而是精准传入下标，减小字符匹检索造成的问题
import json
import numpy as np
import tensorflow as tf
from keras.layers import Input, Lambda, Dense, Average
from keras.models import Model
from keras.callbacks import Callback
from bert4keras.models import build_transformer_model
from bert4keras.backend import K
from bert4keras.optimizers import Adam
from bert4keras.layers import LayerNormalization
from hrconfig import event_extract_train_config
from event_extract.train.utils.utils import seq_gather, get_bert_tokenizer, generate_trained_model_path
from event_extract.train.utils.event_extract_data_util import DataGenerator, get_data
from feedwork.utils import logger as LOG


# 构建默认图和会话
GRAPH = tf.Graph()
SESS = tf.Session(graph=GRAPH)
# 模型配置类
CONFIG = event_extract_train_config.Config()

# 训练后模型保存路径
TRAINED_MODEL_PATH = generate_trained_model_path(CONFIG.trained_model_dir, CONFIG.trained_model_name)
# 获取训练集、验证集
TRAIN_DATA, DEV_DATA = get_data(CONFIG.train_data_path, CONFIG.dev_data_path, CONFIG.supplement_data_dir)

# 获取分字器
TOKENIZER = get_bert_tokenizer(CONFIG.dict_path)


def build_model():
    """
    调用模型参数，搭建事件抽取模型主体，先搭建触发词模型，然后围绕触发词下标搭建其他论元模型。
    :return: 各个论元模型对象
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

            # 搭建模型
            # 动词输入
            trigger_start_in = Input(shape=(None,))
            trigger_end_in = Input(shape=(None,))
            # 动词下标输入
            trigger_index_start_in = Input(shape=(1,))
            trigger_index_end_in = Input(shape=(1,))
            # 宾语输入
            object_start_in = Input(shape=(None,))
            object_end_in = Input(shape=(None,))
            # 主语输入
            subject_start_in = Input(shape=(None,))
            subject_end_in = Input(shape=(None,))
            # 地点输入
            loc_start_in = Input(shape=(None,))
            loc_end_in = Input(shape=(None,))
            # 时间输入
            time_start_in = Input(shape=(None,))
            time_end_in = Input(shape=(None,))
            # 否定词输入
            negative_start_in = Input(shape=(None,))
            negative_end_in = Input(shape=(None,))

            # 动词输入
            trigger_start, trigger_end = trigger_start_in, trigger_end_in
            # 动词下标
            trigger_index_start, trigger_index_end = trigger_index_start_in, trigger_index_end_in
            # 宾语输入
            object_start, object_end = object_start_in, object_end_in
            # 主语输入
            subject_start, subject_end = subject_start_in, subject_end_in
            # 地点输入
            loc_start, loc_end = loc_start_in, loc_end_in
            # 时间输入
            time_start, time_end = time_start_in, time_end_in
            # 否定词输入
            negative_start, negative_end = negative_start_in, negative_end_in

            mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x[0], 2), 0), 'float32'))(bert_model.model.inputs)

            trigger_start_out = Dense(1, activation='sigmoid')(bert_model.model.output)
            trigger_end_out = Dense(1, activation='sigmoid')(bert_model.model.output)
            # 预测trigger动词的模型
            trigger_model = Model(bert_model.model.inputs, [trigger_start_out, trigger_end_out])

            # 动词下标与句子张量融合
            k1v = Lambda(seq_gather)([bert_model.model.output, trigger_index_start])
            k2v = Lambda(seq_gather)([bert_model.model.output, trigger_index_end])
            kv = Average()([k1v, k2v])
            # 融合动词词向量的句子张量
            t = LayerNormalization(conditional=True)([bert_model.model.output, kv])

            # 宾语模型输出
            object_start_out = Dense(1, activation='sigmoid')(t)
            object_end_out = Dense(1, activation='sigmoid')(t)
            # 主语模型输出
            subject_start_out = Dense(1, activation='sigmoid')(t)
            subject_end_out = Dense(1, activation='sigmoid')(t)
            # 地点模型输出
            loc_start_out = Dense(1, activation='sigmoid')(t)
            loc_end_out = Dense(1, activation='sigmoid')(t)
            # 时间模型输出
            time_start_out = Dense(1, activation='sigmoid')(t)
            time_end_out = Dense(1, activation='sigmoid')(t)
            # 否定词模型输出
            negative_start_out = Dense(1, activation='sigmoid')(t)
            negative_end_out = Dense(1, activation='sigmoid')(t)

            object_model = Model(bert_model.model.inputs + [trigger_index_start_in, trigger_index_end_in],
                                 [object_start_out, object_end_out])  # 输入text和trigger，预测object及其关系
            subject_model = Model(bert_model.model.inputs + [trigger_index_start_in, trigger_index_end_in],
                                  [subject_start_out, subject_end_out])  # 输入text和trigger，预测subject及其关系
            loc_model = Model(bert_model.model.inputs + [trigger_index_start_in, trigger_index_end_in],
                              [loc_start_out, loc_end_out])  # 输入text和trigger，预测loc及其关系
            time_model = Model(bert_model.model.inputs + [trigger_index_start_in, trigger_index_end_in],
                               [time_start_out, time_end_out])  # 输入text和trigger，预测time及其关系
            negative_model = Model(bert_model.model.inputs + [trigger_index_start_in, trigger_index_end_in],
                                   [negative_start_out, negative_end_out])  # 否定词模型

            # 主模型
            train_model = Model(
                bert_model.model.inputs + [trigger_start_in, trigger_end_in, trigger_index_start_in,
                                           trigger_index_end_in,
                                           object_start_in, object_end_in, subject_start_in, subject_end_in,
                                           loc_start_in,
                                           loc_end_in, time_start_in, time_end_in, negative_start_in, negative_end_in],
                [trigger_start_out, trigger_end_out, object_start_out, object_end_out, subject_start_out,
                 subject_end_out,
                 loc_start_out, loc_end_out, time_start_out, time_end_out, negative_start_out, negative_end_out])

            # 扩充维度
            trigger_start = K.expand_dims(trigger_start, 2)
            trigger_end = K.expand_dims(trigger_end, 2)
            object_start = K.expand_dims(object_start, 2)
            object_end = K.expand_dims(object_end, 2)
            subject_start = K.expand_dims(subject_start, 2)
            subject_end = K.expand_dims(subject_end, 2)
            loc_start = K.expand_dims(loc_start, 2)
            loc_end = K.expand_dims(loc_end, 2)
            time_start = K.expand_dims(time_start, 2)
            time_end = K.expand_dims(time_end, 2)
            negative_start = K.expand_dims(negative_start, 2)
            negative_end = K.expand_dims(negative_end, 2)

            # 构造模型损失函数
            # 动词损失
            trigger_start_loss = K.binary_crossentropy(trigger_start, trigger_start_out)
            trigger_start_loss = K.sum(trigger_start_loss * mask) / K.sum(mask)
            trigger_end_loss = K.binary_crossentropy(trigger_end, trigger_end_out)
            trigger_end_loss = K.sum(trigger_end_loss * mask) / K.sum(mask)
            # 宾语损失
            object_start_loss = K.sum(K.binary_crossentropy(object_start, object_start_out))
            object_start_loss = K.sum(object_start_loss * mask) / K.sum(mask)
            object_end_loss = K.sum(K.binary_crossentropy(object_end, object_end_out))
            object_end_loss = K.sum(object_end_loss * mask) / K.sum(mask)
            # 主语损失
            subject_start_loss = K.sum(K.binary_crossentropy(subject_start, subject_start_out))
            subject_start_loss = K.sum(subject_start_loss * mask) / K.sum(mask)
            subject_end_loss = K.sum(K.binary_crossentropy(subject_end, subject_end_out))
            subject_end_loss = K.sum(subject_end_loss * mask) / K.sum(mask)
            # 地点损失
            loc_start_loss = K.sum(K.binary_crossentropy(loc_start, loc_start_out))
            loc_start_loss = K.sum(loc_start_loss * mask) / K.sum(mask)
            loc_end_loss = K.sum(K.binary_crossentropy(loc_end, loc_end_out))
            loc_end_loss = K.sum(loc_end_loss * mask) / K.sum(mask)
            # 时间损失
            time_start_loss = K.sum(K.binary_crossentropy(time_start, time_start_out))
            time_start_loss = K.sum(time_start_loss * mask) / K.sum(mask)
            time_end_loss = K.sum(K.binary_crossentropy(time_end, time_end_out))
            time_end_loss = K.sum(time_end_loss * mask) / K.sum(mask)
            # 否定词损失
            negative_start_loss = K.sum(K.binary_crossentropy(negative_start, negative_start_out))
            negative_start_loss = K.sum(negative_start_loss * mask) / K.sum(mask)
            negative_end_loss = K.sum(K.binary_crossentropy(negative_end, negative_end_out))
            negative_end_loss = K.sum(negative_end_loss * mask) / K.sum(mask)

            # 合并损失
            loss = (trigger_start_loss + trigger_end_loss) + (object_start_loss + object_end_loss) + (
                    subject_start_loss + subject_end_loss) + (loc_start_loss + loc_end_loss) + (
                           time_start_loss + time_end_loss) + (negative_start_loss + negative_end_loss)

            train_model.add_loss(loss)
            train_model.compile(optimizer=Adam(CONFIG.learning_rate))
            train_model.summary()

    return trigger_model, subject_model, object_model, time_model, loc_model, negative_model, train_model


# 构建模型
TRIGGER_MODEL, SUBJECT_MODEL, OBJECT_MODEL, TIME_MODEL, LOC_MODEL, NEGATIVE_MODEL, TRAIN_MODEL = build_model()


# 调用模型对传入的数据进行抽取
def extract_items(text_in, **models):
    """
    传入待预测的文本字符串，调用各个模型，句子中的事件论元进行抽取，先抽取动词，然后使用动词下标和句子张量预测其它论元。
    :param text_in: （str）句子字符串
    :return: events(list)
                含有下标的事件各论元组成部分
                        [{ "time": [],
                            "loc": [],
                            "subject": [],
                            "trigger": [],
                            "object": [],
                            "privative": []}]
    :raise:type error
    """
    if not isinstance(text_in, str):
        LOG.error("Type of text_in must be str!")
        raise TypeError

    _tokens = TOKENIZER.tokenize(text_in[:CONFIG.maxlen])
    _text_token_ids, _text_segment_ids = TOKENIZER.encode(first_text=text_in[:CONFIG.maxlen])
    _text_token_ids, _text_segment_ids = np.array([_text_token_ids]), np.array([_text_segment_ids])
    # 抽取事件触发词
    _trigger_start_pre, _trigger_end_pre = TRIGGER_MODEL.predict([_text_token_ids, _text_segment_ids])
    # 动词下标
    _trigger_start_index = np.where(_trigger_start_pre[0] > 0.5)[0]
    _trigger_end_index = np.where(_trigger_end_pre[0] > 0.4)[0]
    _triggers = []
    for i in _trigger_start_index:
        j = _trigger_end_index[_trigger_end_index >= i]
        if len(j) > 0:
            j = j[0]
            _trigger = text_in[i - 1: j]
            _triggers.append((_trigger, i, j))
    # 根据触发词，抽取围绕触发词的各个组成部分
    if _triggers:
        events = []
        _text_token_ids = np.repeat(_text_token_ids, len(_triggers), 0)
        _text_segment_ids = np.repeat(_text_segment_ids, len(_triggers), 0)
        # 动词下标
        _trigger_start_index, _trigger_end_index = np.array([_s[1:] for _s in _triggers]).T.reshape((2, -1, 1))
        # 宾语预测值
        _object_start_pre, _object_end_pre = OBJECT_MODEL.predict(
            [_text_token_ids, _text_segment_ids, _trigger_start_index, _trigger_end_index])  # 宾语
        # 主语预测值
        _subject_start_pre, _subject_end_pre = SUBJECT_MODEL.predict(
            [_text_token_ids, _text_segment_ids, _trigger_start_index, _trigger_end_index])  # 主语
        # 地点预测值
        _loc_start_pre, _loc_end_pre = LOC_MODEL.predict(
            [_text_token_ids, _text_segment_ids, _trigger_start_index, _trigger_end_index])  # 地点
        # 事件预测值
        _time_start_pre, _time_end_pre = TIME_MODEL.predict(
            [_text_token_ids, _text_segment_ids, _trigger_start_index, _trigger_end_index])  # 时间
        # 否定词预测值
        _negative_start_pre, _negative_end_pre = NEGATIVE_MODEL.predict(
            [_text_token_ids, _text_segment_ids, _trigger_start_index, _trigger_end_index])  # 否定词

        for k, _trigger in enumerate(_triggers):
            objects = []
            subjects = []
            locs = []
            times = []
            privatives = []
            # 宾语下标
            _object_start_index = np.where(_object_start_pre[k] > 0.5)[0]
            _object_end_index = np.where(_object_end_pre[k] > 0.4)[0]
            # 主语下标
            _subject_start_index = np.where(_subject_start_pre[k] > 0.5)[0]
            _subject_end_index = np.where(_subject_end_pre[k] > 0.4)[0]
            # 地点下标
            _loc_start_index = np.where(_loc_start_pre[k] > 0.5)[0]
            _loc_end_index = np.where(_loc_end_pre[k] > 0.4)[0]

            # 时间下标
            _time_start_index = np.where(_time_start_pre[k] > 0.5)[0]
            _time_end_index = np.where(_time_end_pre[k] > 0.4)[0]

            # 否定词下标
            _negative_start_index = np.where(_negative_start_pre[k] > 0.5)[0]
            _negative_end_index = np.where(_negative_end_pre[k] > 0.4)[0]

            # 获取宾语
            for o_index in _object_start_index:
                j = _object_end_index[_object_end_index >= o_index]
                if len(j) > 0:
                    j = j[0]
                    _object = text_in[o_index - 1: j]
                    objects.append([_object, [str(o_index - 1), str(j)]])
            # 获取主语
            for s_index in _subject_start_index:
                j = _subject_end_index[_subject_end_index >= s_index]
                if len(j) > 0:
                    j = j[0]
                    _subject = text_in[s_index - 1: j]
                    subjects.append([_subject, [str(s_index - 1), str(j)]])
            # 获取地点
            for l_index in _loc_start_index:
                j = _loc_end_index[_loc_end_index >= l_index]
                if len(j) > 0:
                    j = j[0]
                    _loc = text_in[l_index - 1: l_index]
                    locs.append([_loc, [str(l_index - 1), str(j)]])
            # 获取时间
            for t_index in _time_start_index:
                j = _time_end_index[_time_end_index >= t_index]
                if len(j) > 0:
                    j = j[0]
                    _time = text_in[t_index - 1: j]
                    times.append([_time, [str(t_index - 1), str(j)]])
            # 获取否定词
            for n_index in _negative_start_index:
                j = _negative_end_index[_negative_end_index >= n_index]
                if len(j) > 0:
                    j = j[0]
                    _privative = text_in[n_index - 1: j]
                    privatives.append([_privative, [str(n_index - 1), str(j)]])

            events.append({
                "time": times,
                "loc": locs,
                "subject": subjects,
                "trigger": [[_trigger[0], [str(_trigger[1] - 1), str(_trigger[2])]]],
                "object": objects,
                "privative": privatives
            })

        return events
    else:
        return []


class Evaluate(Callback):
    """
    继承Callback类，改写内部方法,调整训练过程中的学习率，随着训练步数增加，逐渐减小学习率，根据指标保存最优模型。
    """

    def __init__(self, ):
        Callback.__init__(self, )
        self.F1 = []
        self.best = 0.
        self.passed = 0
        self.stage = 0

    def on_batch_begin(self, batch, logs=None):
        """
        在每个批次开始，判断批次步数，来调整训练时的学习率。
        :param batch: 训练批次
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
        f1, precision, recall = self.evaluate()
        self.F1.append(f1)
        if f1 > self.best:
            self.best = f1
            TRAIN_MODEL.save(TRAINED_MODEL_PATH, include_optimizer=True)
        LOG.info(f'f1: {f1}.4f, precision: {precision}.4f, recall: {recall}.4f, best f1: {self.best}.4f\n')

    @staticmethod
    def flat_lists(lists):
        """
        对传入的列表进行拉平
        :param lists: 传入二维列表
        :return: 拉平后的以后列表
        """
        all_elements = []
        for lt in lists:
            all_elements.extend(lt)
        return all_elements

    @staticmethod
    def evaluate():
        """
        衡量元素将关系，计算模型在验证集上的f1\precision\recall
        :return: f1(float), precision(float), recall(float)
        """

        a, b, c = 1e-10, 1e-10, 1e-10

        for d in iter(DEV_DATA):
            extract_res = extract_items(d['sentence'])
            _pred = set()
            _real = set()
            for event in extract_res:
                trigger = event['trigger'][0][0]
                for t in event['time']:
                    _pred.add((trigger, t[0]))
                for l in event['loc']:
                    _pred.add((trigger, l[0]))
                for s in event['subject']:
                    _pred.add((trigger, s[0]))
                for o in event['object']:
                    _pred.add((trigger, o[0]))
            for event in d['events']:
                trigger = event['trigger'][0][0]
                for t in event['time']:
                    _real.add((trigger, t[0]))
                for l in event['loc']:
                    _real.add((trigger, l[0]))
                for s in event['subject']:
                    _real.add((trigger, s[0]))
                for o in event['object']:
                    _real.add((trigger, o[0]))
            a += len(_pred & _real)
            b += len(_pred)
            c += len(_real)

        f1, precision, recall = 2 * a / (b + c), a / b, a / c

        return f1, precision, recall


def model_test(test_data):
    """
    传入测试数据，将测试结果保存到文件中
    :param test_data: (list)测试数据
    :return: None
    :raise: TypeError
    """
    if not isinstance(test_data, list):
        LOG.error("The type of test_data is wrong!")
        raise TypeError

    f = open(CONFIG.pred_path, 'w', encoding='utf-8')
    f.write('[\n')
    first = True
    for d in iter(test_data):
        if not first:
            f.write(',\n')
        else:
            first = False
        events = extract_items(d['sentence'])
        s = json.dumps({
            'sentence': d['sentence'],
            'events': events
        }, ensure_ascii=False)
        f.write(s)
    f.write(']\n')
    f.close()


def model_train():
    """
    进行模型训练
    :return: None
    """
    with SESS.as_default():
        with SESS.graph.as_default():
            # 构造训练数据生成器
            train_d = DataGenerator(TOKENIZER, CONFIG.maxlen, TRAIN_DATA, CONFIG.batch_size)
            # 构造callback模块的评估类
            evaluator = Evaluate()

            if len(TRAIN_DATA) < 5000:
                all_steps = 1000
            else:
                all_steps = train_d.__len__()

            # 模型训练
            TRAIN_MODEL.fit_generator(train_d.__iter__(), steps_per_epoch=all_steps, epochs=CONFIG.epoch,
                                      callbacks=[evaluator])

            model_test(DEV_DATA)
            f1, precision, recall = evaluator.evaluate()
            LOG.info(f"f1:{f1}, precision:{precision}, recall:{recall}")


if __name__ == '__main__':

    model_train()
