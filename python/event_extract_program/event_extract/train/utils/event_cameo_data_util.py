#!/usr/bin/env python
# -*- coding:utf-8 -*-
# 事件cameo模型训练模块所有的数据读取已经生成模块
import numpy as np
from event_extract.train.utils.utils import seq_padding, read_json


def get_data(train_data_path, dev_data_path, label2id_path, id2label_path):
    """
    传入训练集、验证集、标签字典路径，读取json文件内容，训练集、验证集数据以及标签字典。
    :param train_data_path:(str)训练集数据路径
    :param dev_data_path:(str)验证集数据路径
    :param label2id_path:(str)label2id字典路径
    :param id2label_path:(str)id2label字典路径
    :return:train_data(list)、dev_data(list)、label2id(dict)、id2label(dict)
    """
    # 加载训练数据集
    train_data = read_json(train_data_path)
    # 加载验证集
    dev_data = read_json(dev_data_path)
    # 加载标签字典
    label2id = read_json(label2id_path)
    id2label = read_json(id2label_path)

    train_data = [(data, label2id[label]) for data, label in list(train_data.items())]
    dev_data = [(data, label2id[label]) for data, label in list(dev_data.items())]

    return train_data, dev_data, label2id, id2label


class DataGenerator(object):
    """
    构建数据生成器，对传入的数据进行编码、shuffle、分批，迭代返回
    """
    def __init__(self, tokenizer, maxlen, data, batch_size=8, shuffle=True):
        """
        接收数据、批次大小，初始化实体参数。
        :param tokenizer: (object)分字器
        :param maxlen: (int)最大长度
        :param data: (list)数据
        :param batch_size: (int)批量大小
        :param shuffle: (bool)打乱
        """
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        """
        :return: 返回该数据集的步数
        """
        return self.steps

    def __iter__(self):
        """
        构造生成器
        :return: 迭代返回批量数据
        """
        while True:
            # 数据下标
            idxs = list(range(len(self.data)))
            # 对数据进行打乱
            if self.shuffle:
                np.random.shuffle(idxs)
            # 编码ids以及标签
            token_ids, segment_ids, labels = [], [], []
            for i in idxs:
                d = self.data[i]
                # 句子
                text = d[0][:self.maxlen]
                # 使用分字器对句子进行编码
                x1, x2 = self.tokenizer.encode(first_text=text)
                token_ids.append(x1)
                segment_ids.append(x2)
                # 标签值
                y = d[1]
                labels.append(y)
                if len(token_ids) == self.batch_size or i == idxs[-1]:
                    token_ids = seq_padding(token_ids)
                    segment_ids = seq_padding(segment_ids)
                    labels = np.array(labels)
                    yield [token_ids, segment_ids], labels
                    token_ids, segment_ids, labels = [], [], []
