#!/usr/bin/env python
# -*- coding:utf-8 -*-


class Config(object):
    """
    构建参数类，传递代码所需的所有参数。
    """

    def __init__(self):

        # 模型相关参数
        self.model_type = "bert"
        # 模型参数
        self.config_path = 'event_extract/train/model/initial_model/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_config.json'
        # 初始化模型路径
        self.checkpoint_path = 'event_extract/train/model/initial_model/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_model.ckpt'
        # 模型字典路径
        self.dict_path = 'event_extract/train/model/initial_model/chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt'
        # 训练后模型保存路径
        self.trained_model_dir = "event_extract/train/model/event_extract_trained_model"
        # 训练后模型名称
        self.trained_model_name = "extract_bert4keras_v03_model.h5"
        # 训练集数据保存路径
        self.train_data_path = "event_extract/train/resources/event_extract_data/raw_data/train_data.json"
        # 测试集数据保存路径
        self.dev_data_path = "event_extract/train/resources/event_extract_data/raw_data/dev_data.json"
        # 补充数据保存文件夹
        self.supplement_data_dir = "event_extract/train/resources/event_extract_data/supplement"
        # 训练后测试集预测结果
        self.pred_path = 'event_extract/train/resources/event_extract_data/test_pred.json'
        # 训练集测试集划分比例
        self.ratio = 8
        # 训练批次大小
        self.batch_size = 8
        # 循环
        self.epoch = 100
        # 学习率
        self.learning_rate = 5e-5
        # 最小学习率
        self.min_learning_rate = 1e-5
        # 字符串最大长度
        self.maxlen = 160
