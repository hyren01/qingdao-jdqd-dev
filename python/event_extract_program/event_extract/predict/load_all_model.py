#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
该模块加载事件抽取过程使用的所有模型，并返回模型对象，以及对数据预测值进行处理返回给flask模块
"""
import numpy as np
from keras.layers import Input, Lambda, Dense, Average
from keras.models import Model
from bert4keras.models import build_transformer_model
from bert4keras.layers import LayerNormalization
from pyhanlp import HanLP
from event_extract.predict.model_utils import seq_gather, get_bert_tokenizer, read_json
from hrconfig import event_extract_predict_config
from feedwork.utils import logger as LOG

# 参数类
CONFIG = event_extract_predict_config.Config()

# 使用bert字典，构建tokenizer类
TOKENIZER = get_bert_tokenizer(CONFIG.dict_path)

# 事件状态与id字典
STATE2ID = {'happened': 0, 'happening': 1, 'possible': 2}
ID2STATE = {0: 'happened', 1: 'happening', 2: 'possible'}
# 事件cameo字典
ID2CAMEO = read_json(CONFIG.id2cameo_path)


# 加载事件抽取模型
def get_extract_model():
    """
    构建事件抽取模型结构，加载模型参数，返回模型对象
    :return: 各个部分的模型对象
    """
    # 构建bert模型主体
    bert_model = build_transformer_model(
        config_path=CONFIG.bert_config_path,
        return_keras_model=False,
        model=CONFIG.model_type
    )

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

    trigger_index_start, trigger_index_end = trigger_index_start_in, trigger_index_end_in

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
        bert_model.model.inputs + [trigger_start_in, trigger_end_in, trigger_index_start_in, trigger_index_end_in,
                                   object_start_in, object_end_in, subject_start_in, subject_end_in, loc_start_in,
                                   loc_end_in, time_start_in, time_end_in, negative_start_in, negative_end_in],
        [trigger_start_out, trigger_end_out, object_start_out, object_end_out, subject_start_out, subject_end_out,
         loc_start_out, loc_end_out, time_start_out, time_end_out, negative_start_out, negative_end_out])

    train_model.load_weights(CONFIG.event_extract_model_path)

    return trigger_model, object_model, subject_model, loc_model, time_model, negative_model


# 加载事件时态判断模型
def get_state_model():
    """
    构建事件状态模型，加载模型参数，返回模型对象
    :return: state_model
    """
    # 构建bert模型主体
    bert_model = build_transformer_model(
        config_path=CONFIG.bert_config_path,
        return_keras_model=False,
        model=CONFIG.model_type
    )

    trigger_start_index = Input(shape=(1,))
    trigger_end_index = Input(shape=(1,))

    k1v = Lambda(seq_gather)([bert_model.model.output, trigger_start_index])
    k2v = Lambda(seq_gather)([bert_model.model.output, trigger_end_index])
    kv = Average()([k1v, k2v])
    t = LayerNormalization(conditional=True)([bert_model.model.output, kv])
    t = Lambda(lambda x: x[:, 0])(t)  # 取出[CLS]对应的向量用来做分类

    state_out_put = Dense(3, activation='softmax')(t)
    state_model = Model(bert_model.model.inputs + [trigger_start_index, trigger_end_index], state_out_put)

    # 加载模型
    state_model.load_weights(CONFIG.event_state_model_path)

    return state_model


def get_cameo_model():
    """
    加载事件类别（CAMEO）模型，返回事件类别模型对象
    :return: 事件类型模型

    """
    # 搭建bert模型主体
    bert_model = build_transformer_model(
        config_path=CONFIG.bert_config_path,
        return_keras_model=False,
        model=CONFIG.model_type
    )

    t = Lambda(lambda x: x[:, 0])(bert_model.model.output)  # 取出[CLS]对应的向量用来做分类
    cameo_out_put = Dense(len(ID2CAMEO), activation='softmax')(t)

    cameo_model = Model(bert_model.model.inputs, cameo_out_put)
    # 加载模型参数
    cameo_model.load_weights(CONFIG.event_cameo_model_path)

    return cameo_model


def extract_items(text_in, state_model, trigger_model, object_model, subject_model, loc_model, time_model,
                  negative_model):
    """
    传入待抽取事件的句子，抽取事件的各个模型，对事件句子中的事件进行抽取。
    :param text_in: (str)待抽取事件的句子
    :param state_model: 事件状态模型
    :param trigger_model: 事件触发词模型
    :param object_model: 事件宾语模型
    :param subject_model: 事件主语模型
    :param loc_model: 事件地点模型
    :param time_model: 事件时间模型
    :param negative_model: 事件否定词模型
    :return: events(list)事件列表
    """
    if not isinstance(text_in, str):
        LOG.error("待抽取的句子不是字符串！")
        raise ValueError
    # 使用bert分字器对字符串进行编码
    _t1, _t2 = TOKENIZER.encode(first_text=text_in[:CONFIG.maxlen])
    _t1, _t2 = np.array([_t1]), np.array([_t2])
    # 动词预测值
    _k1, _k2 = trigger_model.predict([_t1, _t2])
    # 动词下标
    _k1, _k2 = np.where(_k1[0] > 0.5)[0], np.where(_k2[0] > 0.4)[0]
    _triggers = []
    for i in _k1:
        j = _k2[_k2 >= i]
        if len(j) > 0:
            j = j[0]
            _trigger = text_in[i - 1: j]
            # 事件状态
            state = state_model.predict([_t1, _t2, np.array([i], dtype=np.int32), np.array([j], dtype=np.int32)])
            state = np.argmax(state, axis=1)
            state = ID2STATE[state[0]]
            _triggers.append((_trigger, i, j, state))

    if _triggers:
        events = []
        # 构造预测事件论元的字符串编码集
        _t1 = np.repeat(_t1, len(_triggers), 0)
        _t2 = np.repeat(_t2, len(_triggers), 0)
        # 动词下标
        _k1, _k2 = np.array([_s[1:3] for _s in _triggers]).T.reshape((2, -1, 1))
        # 宾语预测值
        _o1, _o2 = object_model.predict([_t1, _t2, _k1, _k2])
        # 主语预测值
        _s1, _s2 = subject_model.predict([_t1, _t2, _k1, _k2])
        # 地点预测值
        _l1, _l2 = loc_model.predict([_t1, _t2, _k1, _k2])
        # 时间预测值
        _tm1, _tm2 = time_model.predict([_t1, _t2, _k1, _k2])
        # 否定词预测值
        _n1, _n2 = negative_model.predict([_t1, _t2, _k1, _k2])

        for i, _trigger in enumerate(_triggers):
            objects = []
            subjects = []
            locs = []
            times = []
            negatives = []
            # 宾语下标
            _oo1, _oo2 = np.where(_o1[i] > 0.5)[0], np.where(_o2[i] > 0.4)[0]
            # 主语下标
            _so1, _so2 = np.where(_s1[i] > 0.5)[0], np.where(_s2[i] > 0.4)[0]
            # 地点下标
            _lo1, _lo2 = np.where(_l1[i] > 0.5)[0], np.where(_l2[i] > 0.4)[0]
            # 时间下标
            _tmo1, _tmo2 = np.where(_tm1[i] > 0.5)[0], np.where(_tm2[i] > 0.4)[0]
            # 否定词下标
            _no1, _no2 = np.where(_n1[i] > 0.5)[0], np.where(_n2[i] > 0.4)[0]
            # 按照下标，在字符串中索引对应的论元
            for i in _oo1:
                j = _oo2[_oo2 >= i]
                if len(j) > 0:
                    j = j[0]
                    _object = text_in[i - 1: j]
                    objects.append(_object)
            for i in _so1:
                j = _so2[_so2 >= i]
                if len(j) > 0:
                    j = j[0]
                    _subject = text_in[i - 1: j]
                    subjects.append(_subject)
            for i in _lo1:
                j = _lo2[_lo2 >= i]
                if len(j) > 0:
                    j = j[0]
                    _loc = text_in[i - 1: j]
                    locs.append(_loc)
            for i in _tmo1:
                j = _tmo2[_tmo2 >= i]
                if len(j) > 0:
                    j = j[0]
                    _time = text_in[i - 1: j]
                    times.append(_time)
            for i in _no1:
                j = _no2[_no2 >= i]
                if len(j) > 0:
                    j = j[0]
                    _negative = text_in[i - 1: j]
                    negatives.append(_negative)

            events.append({
                "event_datetime": times[0] if times else "",
                "event_location": locs[0] if locs else "",
                "subject": ",".join(subjects) if subjects else "",
                "verb": _trigger[0],
                "object": ",".join(objects) if objects else "",
                "negative_word": negatives[0] if negatives else "",
                "state": _trigger[3],
                "triggerloc_index": [int(_trigger[1] - 1), int(_trigger[2] - 1)]
            })
        # {'event_datetime': "", 'event_location': "", 'subject': '美国',
        # 'verb': '采取', 'object': '数十次由总统承诺停止的联合演习',
        # 'negative_word': "取消", 'state': 'possible', triggerloc_index:[12,15]}
        return events
    else:
        return []


def get_ners(events):
    """
    传入事件列表，使用hanlp抽取事件中的实体名词
    :param events: (list)事件列表
    :return: events(list)补充实体成分后的事件字典列表
    """
    if not isinstance(events, list):
        LOG.error("用于抽取实体的事件列表格式不对！")
        raise TypeError

    for event in events:
        sentence = ''.join([event[i] for i in event.keys() if i != 'state' and i != 'triggerloc_index'])
        words = HanLP.segment(sentence)
        event["namedentity"] = {'organization': [], 'location': [], 'person': []}

        for once in words:

            if str(once.nature).startswith("ns"):
                if str(once.word) in event["namedentity"]["location"]:
                    pass
                else:
                    event["namedentity"]["location"].append(str(once.word))

            elif str(once.nature).startswith("nr"):
                if str(once.word) in event["namedentity"]["person"]:
                    pass
                else:
                    event["namedentity"]["person"].append(str(once.word))

            elif str(once.nature).startswith("nt"):
                if str(once.word) in event["namedentity"]["organization"]:
                    pass
                else:
                    event["namedentity"]["organization"].append(str(once.word))

    return events


def get_event_cameo(cameo_model, events):
    """
    传入事件cameo模型和事件列表，判断每个事件的cameo，并添加到事件列表中
    :param cameo_model:事件cameo模型
    :param events:(list) 事件列表
    :return:events(list)事件列表
    """
    for event in events:
        short_sentence = "".join([event["subject"], event["negative_word"], event["verb"], event["object"]])
        _t1, _t2 = TOKENIZER.encode(first_text=short_sentence)
        _t1, _t2 = np.array([_t1]), np.array([_t2])
        event_cameo = cameo_model.predict([_t1, _t2])
        event_cameo = np.argmax(event_cameo, axis=1)
        event_id = event_cameo[0]
        event["cameo"] = ID2CAMEO[f"{event_id}"]

    return events


if __name__ == '__main__':
    state_model = get_state_model()
    trigger_model, object_model, subject_model, loc_model, time_model, privative_model = get_extract_model()
    cameo_model = get_cameo_model()
    # while True:
    #     sentence = input("请输入想要预测的句子。")
    #     R = extract_items(sentence, state_model, trigger_model, object_model, subject_model, loc_model, time_model,
    #                       privative_model)
    #     print(R)
    #
    #     a = get_event_cameo(cameo_model, R)
