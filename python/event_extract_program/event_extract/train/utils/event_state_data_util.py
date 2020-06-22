#!/usr/bin/env python
# -*- coding:utf-8 -*-
# 事件cameo模型训练模块所有的数据读取已经生成模块
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from event_extract.train.utils.utils import seq_padding, list_find, read_json


def get_data(data_path):
    """
    传入数据路径，对数据进行解析并划分训练集测试集，返回训练集、测试集和标签字典
    :param data_path:(str)数据路径
    :return:train_data_list(ndarrary), test_data_list(ndarrary), test_df(dataframe), state2id(dict), id2state(dict)
    """

    # 模型状态 下标 字典
    state2id = {'happened': 0, 'happening': 1, 'possible': 2}
    id2state = {0: 'happened', 1: 'happening', 2: 'possible'}

    # 加载训练数据
    total_data = read_json(data_path)

    # 原始语句
    data_list = []
    for enery_data in total_data:

        sen = enery_data['sentence']
        for event in enery_data['events']:
            if event['trigger'] and event['state']:
                trig = event['trigger'][0]
                stat = event['state'][0]
                stat_id = state2id[stat]
                data_list.append((sen, trig, stat_id))

    data = np.array(data_list)
    np.random.shuffle(data)  # 随机打乱
    # 取前80%为训练集
    train_data = data[:int(0.8 * len(data))]
    # 将np.array转为dataframe，并对两列赋列名
    train_df = pd.DataFrame(train_data, columns=['sentences', 'trig', 'la'])

    # 剩余百分之20为测试集
    test_data = data[int(0.8 * len(data)):]
    test_df = pd.DataFrame(test_data, columns=['sentences', 'trig', 'la'])

    # 训练数据、测试数据和标签转化为模型输入格式
    train_data_list = []
    for data_row in train_df.iloc[:].itertuples():
        train_data_list.append((data_row.sentences, data_row.trig, to_categorical(data_row.la, 3)))
    train_data_list = np.array(train_data_list)

    # 验证集
    test_data_list = []
    for data_row in test_df.iloc[:].itertuples():
        test_data_list.append((data_row.sentences, data_row.trig, to_categorical(0, 3)))
    test_data_list = np.array(test_data_list)

    return train_data_list, test_data_list, test_df, state2id, id2state


class DataGenerator(object):
    """
    构建数据生成器，对传入的数据进行编码、shuffle、分批，迭代返回。
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
            idxs = list(range(len(self.data)))
            if self.shuffle:
                np.random.shuffle(idxs)  # 对传入的数据进行打乱

            # 初始化数据列表
            text_token_ids, text_segment_ids, trigger_start_index, trigger_end_index, labels = [], [], [], [], []
            for i in idxs:
                d = self.data[i]
                text = d[0][:self.maxlen]
                tokens = self.tokenizer.tokenize(text)
                trigger_token = self.tokenizer.tokenize(d[1])[1:-1]
                triggerid = list_find(tokens, trigger_token)
                key = (triggerid, triggerid + len(trigger_token))
                k1 = key[0]
                k2 = key[1]
                trigger_start_index.append([k1])
                trigger_end_index.append([k2 - 1])
                x1, x2 = self.tokenizer.encode(first_text=text)
                text_token_ids.append(x1)
                text_segment_ids.append(x2)
                y = d[2]
                labels.append(y)
                # 如果数据量达到批次大小或最后一个批次就进行填充并迭代出去
                if len(text_token_ids) == self.batch_size or i == idxs[-1]:
                    # 序列填充
                    text_token_ids = seq_padding(text_token_ids)
                    text_segment_ids = seq_padding(text_segment_ids)
                    trigger_start_index, trigger_end_index = np.array(trigger_start_index), np.array(trigger_end_index)
                    labels = np.array(labels)
                    yield [text_token_ids, text_segment_ids, trigger_start_index, trigger_end_index], labels
                    # 重现将数据各部分列表置空
                    text_token_ids, text_segment_ids = [], []
                    trigger_start_index, trigger_end_index = [], []
                    labels = []
