#!/usr/bin/env python
# -*- coding:utf-8 -*-
# 模型训练数据处理公用部分
import numpy as np
import codecs
import json
import tensorflow as tf
from bert4keras.tokenizers import Tokenizer
from bert4keras.backend import K


def read_json(file_path):
    """
    传入json文件路径，读取文件内容，返回json解析后的数据。
    :param file_path: (str)json文件路径
    :return: data
    :raise: ValueError
    """
    if not file_path.endswith(".json"):
        raise ValueError
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
    except TypeError:
        with open(file_path, "r", encoding="gbk") as file:
            content = file.read()

    data = json.loads(content)

    return data


def get_token_dict(dict_path):
    """
    传入bert字典路径，读取字典内容，返回字典
    :param dict_path: (str)字典路径
    :return: token_dict(dict)模型字典
    """
    if not isinstance(dict_path, str):
        raise ValueError

    token_dict = {}
    # 加载模型对应的字典
    with codecs.open(dict_path, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)

    return token_dict


class OurTokenizer(Tokenizer):
    """
    改写bert自带的分字类，处理空白符和不识别字符的分字问题。
    """
    def _tokenize(self, text):
        """
        对传入的字符串进行分字
        :param text: (str)字符串
        :return:token_list(list)分字列表
        """
        token_list = []
        for c in text:
            if c in self._token_dict:
                token_list.append(c)
            elif self._is_space(c):
                token_list.append('[unused1]')  # space类用未经训练的[unused1]表示
            else:
                token_list.append('[UNK]')  # 剩余的字符是[UNK]
        return token_list


def get_bert_tokenizer(dict_path):
    """
    传入字典路径，读取字典内容，返回分字器
    :param dict_path: (str)字典路径
    :return: tokenizer(object)分字器
    """
    token_dict = get_token_dict(dict_path)
    # 构建tokenizer类
    tokenizer = OurTokenizer(token_dict)

    return tokenizer


def seq_padding(seq, padding=0):
    """
    对传入的序列进行填充
    :param seq: (list)传入的待填充的序列
    :param padding: 填充字符
    :return: 返回填充后的数组数据
    """
    length = [len(x) for x in seq]
    max_len = max(length)
    return np.array([
        np.concatenate([x, [padding] * (max_len - len(x))]) if len(x) < max_len else x for x in seq
    ], dtype=np.float32)


def seq_gather(x):
    """
    传入seq idxs, 对下标对应的张量与原始张量seq进行融合，
    seq是[None, seq_len, s_size]的格式，
    idxs是[None, 1]的格式，在seq的第i个序列中选出第idxs[i]个向量，
    最终输出[None, s_size]的向量。
    :param x: （seq, idxs）
    :return: 融合后的张量
    """
    seq, idxs = x
    idxs = K.cast(idxs, 'int32')
    batch_idxs = K.arange(0, K.shape(seq)[0])
    batch_idxs = K.expand_dims(batch_idxs, 1)
    idxs = K.concatenate([batch_idxs, idxs], 1)
    return tf.gather_nd(seq, idxs)


def list_find(list1, list2):
    """
    在序列list1中寻找紫川list2,如果找到，返回第一个下标；
    如果找不到，返回-1
    :param list1: (list, str)
    :param list2: (list, str)
    :return: i(int)起始下标， -1 未找到
    """
    n_list2 = len(list2)
    for i in range(len(list1)):
        if list1[i: i + n_list2] == list2:
            return i
    return -1