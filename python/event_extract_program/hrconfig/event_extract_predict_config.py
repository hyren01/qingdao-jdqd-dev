#!/usr/bin/env python
# coding:utf-8


class Config(object):
    """
    构建参数类，传递代码运行所需的所有参数
    """
    def __init__(self, ):
        # 模型类型
        self.model_type = "bert"
        # 字符串最大长度
        self.maxlen = 160
        # 事件抽取模型路径
        self.event_extract_model_path = "event_extract/resources/model/extract_bert4keras_v03_model.h5"
        # 事件状态判断模型路径
        self.event_state_model_path = "event_extract/resources/model/state_bert4keras_v03_model.h5"
        # 事件类别模型路径
        self.event_cameo_model_path = "event_extract/resources/model/type_bert4keras_v03_model.h5"
        # bert模型参数json文件路径
        self.bert_config_path = "event_extract/resources/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_config.json"
        # bert模型字典保存的路径
        self.dict_path = "event_extract/resources/chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt"
        # 小牛翻译的key
        self.user_key = "dfc242105d925bf99d82bc16c2171a00"
        # 小牛翻译的网址
        self.translate_url = "http://api.niutrans.vip/NiuTransServer/translation?"
        # 事件类型字典文件
        self.id2cameo_path = "event_extract/resources/id2label.json"
