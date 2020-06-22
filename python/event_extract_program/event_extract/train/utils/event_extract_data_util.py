#!/usr/bin/env python
# -*- coding:utf-8 -*-
# 事件抽取模型训练模块所有的数据读取已经生成模块
import os
from random import choice
import numpy as np
from event_extract.train.utils.utils import seq_padding, read_json


def get_data(train_data_path, dev_data_path, supplement_data_dir):
    """
    传入训练集、验证集、补充数据路径，读取并解析json数据，将补充数据补充到训练集中，返回解析后的数据
    :param train_data_path:(str)训练集路径
    :param dev_data_path:(str)验证集路径
    :param supplement_data_dir:(str)补充数据保存路径
    :return:train_data(list), dev_data(list)
    """
    # 加载训练数据集
    train_data = read_json(train_data_path)
    # 加载验证集
    dev_data = read_json(dev_data_path)

    # 加载补充数据
    file_list = os.listdir(supplement_data_dir)
    supplement_data = []
    for file in file_list:
        supplement_data_path = os.path.join(supplement_data_dir, file)
        supplement_data.extend(read_json(supplement_data_path))
    train_data.extend(supplement_data)

    return train_data, dev_data


class DataGenerator(object):
    """
    构建数据生成器，对传入的数据进行编码、shuffle、分批，迭代返回
    """

    def __init__(self, tokenizer, maxlen, data, batch_size=8):
        """
        接收分字器、最大长度、数据、批量大小
        :param tokenizer: (object)分字器
        :param maxlen: (int)最大长度
        :param data: (list)数据
        :param batch_size: (int)批量大小
        """
        self.data = data
        self.maxlen = maxlen
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        """
        :return:返回该数据集的步数
        """
        return self.steps

    def __iter__(self):
        """
        构造生成器
        :return: 迭代返回批量数据
        """
        while True:
            idxs = list(range(len(self.data)))
            np.random.shuffle(idxs)  # 对传入的数据进行打乱

            # 初始化数据列表
            # 字符串编码
            text_token_ids, text_segment_ids = [], []
            # 动词标签值编码
            trigger_start_label, trigger_end_label = [], []
            # 动词下标
            trigger_start_index, trigger_end_index = [], []
            # 宾语标签值编码
            object_start_label, object_end_label = [], []
            # 主语标签值编码
            subject_start_label, subject_end_label = [], []
            # 地点标签值编码
            loc_start_label, loc_end_label = [], []
            # 时间标签值编码
            time_start_label, time_end_label = [], []
            # 否定词标签值编码
            negative_start_label, negative_end_label = [], []

            # 遍历下标打乱后的下标列表
            for i in idxs:
                d = self.data[i]
                text = d['sentence'][:self.maxlen]
                tokens = self.tokenizer.tokenize(text)
                items = {}

                # 根据得到的触发词，抽取相应的论元组成部分
                for event in d['events']:
                    trigger_token = self.tokenizer.tokenize(event['trigger'][0][0])[1:-1]
                    triggerid = int(event['trigger'][0][1][0]) + 1
                    key = (triggerid, triggerid + len(trigger_token))
                    if key not in items:
                        items[key] = []

                    # 初始化主体、客体、地点、时间、否定词下标列表
                    subject_ids, object_ids, loc_ids, time_ids, privative_ids = [], [], [], [], []

                    if event['subject']:  # 主语
                        for p in event['subject']:
                            subject_token = self.tokenizer.tokenize(p[0])[1:-1]
                            subject_id = int(p[1][0]) + 1
                            subject_ids.append((subject_id, subject_id + len(subject_token)))

                    if event['object']:  # 宾语
                        for o in event['object']:
                            object_token = self.tokenizer.tokenize(o[0])[1:-1]
                            object_id = int(o[1][0]) + 1
                            object_ids.append((object_id, object_id + len(object_token)))

                    if event['loc']:  # 地点
                        for l in event['loc']:
                            loc_token = self.tokenizer.tokenize(l[0])[1:-1]
                            loc_id = int(l[1][0]) + 1
                            loc_ids.append((loc_id, loc_id + len(loc_token)))

                    if event['time']:  # 时间
                        for t in event['time']:
                            time_token = self.tokenizer.tokenize(t[0])[1:-1]
                            time_id = int(t[1][0]) + 1
                            time_ids.append((time_id, time_id + len(time_token)))

                    if event['privative']:  # 否定词
                        for n in event['privative']:
                            privative_token = self.tokenizer.tokenize(n[0])[1:-1]
                            privative_id = int(n[1][0]) + 1
                            privative_ids.append((privative_id, privative_id + len(privative_token)))
                    # 将所有的组成部分以触发词下标为键构成字典
                    items[key].append((subject_ids, object_ids, loc_ids, time_ids, privative_ids))

                if items:
                    t1, t2 = self.tokenizer.encode(first_text=text)
                    text_token_ids.append(t1)
                    text_segment_ids.append(t2)
                    d1, d2 = np.zeros(len(tokens)), np.zeros(len(tokens))
                    for j in items:
                        try:
                            d1[j[0]] = 1
                            d2[j[1] - 1] = 1
                        except IndexError:
                            d1, d2 = np.zeros(len(tokens)), np.zeros(len(tokens))
                            continue
                    k1, k2 = np.array(list(items.keys())).T
                    k1 = choice(k1)
                    k2 = choice(k2[k2 >= k1])
                    o1, o2 = np.zeros((len(tokens))), np.zeros((len(tokens)))
                    p1, p2 = np.zeros((len(tokens))), np.zeros((len(tokens)))
                    l1, l2 = np.zeros((len(tokens))), np.zeros((len(tokens)))
                    tm1, tm2 = np.zeros((len(tokens))), np.zeros((len(tokens)))
                    n1, n2 = np.zeros((len(tokens))), np.zeros((len(tokens)))
                    for e in items.get((k1, k2), []):
                        try:
                            for j in e[0]:  # 主语
                                p1[j[0]] = 1
                                p2[j[1] - 1] = 1
                            for j in e[1]:  # 宾语
                                o1[j[0]] = 1
                                o2[j[1] - 1] = 1
                            for j in e[2]:  # 地点
                                l1[j[0]] = 1
                                l2[j[1] - 1] = 1
                            for j in e[3]:  # 时间
                                tm1[j[0]] = 1
                                tm2[j[1] - 1] = 1
                            for j in e[4]:  # 否定词
                                n1[j[0]] = 1
                                n2[j[1] - 1] = 1
                        except IndexError:
                            o1, o2 = np.zeros((len(tokens))), np.zeros((len(tokens)))
                            p1, p2 = np.zeros((len(tokens))), np.zeros((len(tokens)))
                            l1, l2 = np.zeros((len(tokens))), np.zeros((len(tokens)))
                            tm1, tm2 = np.zeros((len(tokens))), np.zeros((len(tokens)))
                            n1, n2 = np.zeros((len(tokens))), np.zeros((len(tokens)))
                            continue

                    trigger_start_label.append(d1)
                    trigger_end_label.append(d2)
                    trigger_start_index.append([k1])
                    trigger_end_index.append([k2 - 1])
                    object_start_label.append(o1)
                    object_end_label.append(o2)
                    subject_start_label.append(p1)
                    subject_end_label.append(p2)
                    loc_start_label.append(l1)
                    loc_end_label.append(l2)
                    time_start_label.append(tm1)
                    time_end_label.append(tm2)
                    negative_start_label.append(n1)
                    negative_end_label.append(n2)
                    # 如果数据量达到批次大小或最后一个批次就进行填充并迭代出去
                    if len(text_token_ids) == self.batch_size or i == idxs[-1]:
                        # 序列填充
                        text_token_ids = seq_padding(text_token_ids)  # 原始句子编码
                        text_segment_ids = seq_padding(text_segment_ids)
                        trigger_start_label = seq_padding(trigger_start_label)  # 动词标签
                        trigger_end_label = seq_padding(trigger_end_label)
                        object_start_label = seq_padding(object_start_label)  # 宾语标签
                        object_end_label = seq_padding(object_end_label)
                        subject_start_label = seq_padding(subject_start_label)  # 主语标签
                        subject_end_label = seq_padding(subject_end_label)
                        loc_start_label = seq_padding(loc_start_label)  # 地点标签
                        loc_end_label = seq_padding(loc_end_label)
                        time_start_label = seq_padding(time_start_label)  # 时间标签
                        time_end_label = seq_padding(time_end_label)
                        negative_start_label = seq_padding(negative_start_label)  # 否定词标签
                        negative_end_label = seq_padding(negative_end_label)
                        trigger_start_index, trigger_end_index = np.array(trigger_start_index), np.array(
                            trigger_end_index)  # 动词下标

                        yield [text_token_ids, text_segment_ids, trigger_start_label, trigger_end_label,
                               trigger_start_index, trigger_end_index, object_start_label, object_end_label,
                               subject_start_label, subject_end_label, loc_start_label, loc_end_label, time_start_label,
                               time_end_label, negative_start_label, negative_end_label], None
                        # 重现将数据各部分列表置空
                        text_token_ids, text_segment_ids = [], []
                        trigger_start_label, trigger_end_label = [], []
                        trigger_start_index, trigger_end_index = [], []
                        object_start_label, object_end_label = [], []
                        subject_start_label, subject_end_label = [], []
                        loc_start_label, loc_end_label = [], []
                        time_start_label, time_end_label = [], []
                        negative_start_label, negative_end_label = [], []
