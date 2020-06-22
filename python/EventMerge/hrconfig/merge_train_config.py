#!/usr/bin/env python
# coding : utf-8


class Config(object):
    """
    创建参数类，供所有模块的初始化调用。
    """

    def __init__(self):

        # 批量大小
        self.batch_size = 8
        # 最大最小学习率
        self.learning_rate = 5e-5
        self.min_learning_rate = 1e-5
        # 模型初始路径
        self.bert_model_path = "EventMerge/train/model/initial_model/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_model.ckpt"
        # 模型参数
        self.bert_config = "EventMerge/train/model/initial_model/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_config.json"
        # 字典
        self.vocab = "EventMerge/train/model/initial_model/chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt"
        # 训练集路径
        self.train_data_path = "EventMerge/train/resources/data/datav06/train.txt"
        # 验证集路径
        self.dev_data_path = "EventMerge/train/resources/data/datav06/dev.txt"
        # 测试集路径
        self.test_data_path = "EventMerge/train/resources/data/datav06/test.txt"
        # 训练后的模型路径
        self.trained_model_path = "EventMerge/train/model/match_model.h5"
        # 转化后的向量路径
        self.vector_data_path = "EventMerge/train/resources/output/vector_data"
        # 存储转化样本的字典
        self.vector_id_dict_dir = "EventMerge/train/resources/output"
