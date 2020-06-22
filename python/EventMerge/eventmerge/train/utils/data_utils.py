#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Mr Fan
# @Time: 2020年05月09
import os
import json
import numpy as np
from tqdm import tqdm


def load_data(file_path):
    """
    传入文件路径，按行读取文件内容，去除换行符，返回数据列表
    :param file_path: (str)数据保存路径
    :return: data(list)数据列表
    :raise:TypeError
    """
    if not isinstance(file_path, str):
        raise TypeError
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = f.readlines()
    except TypeError:
        with open(file_path, "r", encoding="gbk") as f:
            data = f.readlines()

    data = [once.replace("\n", "") for once in data if once]

    return data


def generate_vec(mode, data, tokenizer, bert_model, vector_data_dir, vector_id_dict_dir):
    """
    传入保存模式、数据、分词器、模型对象、向量保存文件夹路径、
    :param mode: （str）数据模式 train dev test
    :param data: (list) 数据列表 sentence sentence2 label
    :param tokenizer: 分字器
    :param bert_model: bert模型对象
    :param vector_data_dir: （str）向量保存文件夹
    :param vector_id_dict_dir: (str)事件{cameo:[id]}保存文件夹
    :return: None
    """
    # 下标列表
    idxs = list(range(len(data)))
    # 数据字典
    data_dict = {}
    # 保存向量的列表
    X = []

    for j, i in tqdm(enumerate(idxs)):
        d = data[i]
        text_01 = d.split("	")[0]
        text_02 = d.split("	")[1]

        data_dict.setdefault(j // 10000, [])

        if text_01 not in data_dict[j // 10000]:
            data_dict[j // 10000].append(text_01)

            x1_1, x1_2 = tokenizer.encode(first_text=text_01)

            x1 = bert_model.model.predict([np.array([x1_1]), np.array([x1_2])])[0][0]

            X.append(x1)

        if text_02 not in data_dict[j // 10000]:
            data_dict[j // 10000].append(text_02)
            x2_1, x2_2 = tokenizer.encode(first_text=text_02)
            x2 = bert_model.model.predict([np.array([x2_1]), np.array([x2_2])])[0][0]

            X.append(x2)

        if (j + 1) % 10000 == 0 or (j + 1) == len(data):
            X = np.array(X)
            file_name = "{}_{}.npy".format(mode, j // 10000)
            file_path = os.path.join(vector_data_dir, file_name)
            np.save(file_path, X)
            X = []

    dict_name = "{}_dict.json".format(mode)
    dict_path = os.path.join(vector_id_dict_dir, dict_name)

    with open(dict_path, "w", encoding="utf-8") as f:
        content = json.dumps(data_dict, ensure_ascii=False, indent=4)
        f.write(content)


def load_vec_data(vector_data_dir):
    """
    传入向量数据保存的文件夹，加载所有的向量数据到内存中。
    :param vector_data_dir: (str)向量数据保存的文件夹
    :return: vec_data(dict)向量数据字典
    """
    # 将所有向量加载到内存中
    vec_data = {}
    file_list = os.listdir(vector_data_dir)
    for file in file_list:
        vec_data["{}".format(file)] = np.load(os.path.join(vector_data_dir, file))

    return vec_data


# 构架数据生成类
class DataGenerator(object):
    """
    构建数据生成类，传入数据，实现向量数据迭代输出。
    """

    def __init__(self, mode, data, vec_data, vector_id_dict_dir, batch_size, shuffle=True):
        self.mode = mode
        self.data = data
        self.vec_data = vec_data
        self.vector_id_dict_dir = vector_id_dict_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

        dict_name = f"{self.mode}_dict.json"
        dict_path = os.path.join(vector_id_dict_dir, dict_name)

        with open(dict_path, "r", encoding="utf-8") as f:
            self.data_dict = f.read()
            self.data_dict = json.loads(self.data_dict)

    def __len__(self):
        """
        :return:返回数据集步数
        """
        return self.steps

    def __iter__(self):

        while True:
            idxs = list(range(len(self.data)))
            if self.shuffle:
                np.random.shuffle(idxs)
            vectors_1, vectors_2, labels = [], [], []
            for i in idxs:
                x1 = self.data[i].split("\t")[0]
                x2 = self.data[i].split("\t")[1]
                y = int(self.data[i].split("\t")[2])

                key = str(i // 10000)
                x1_id = self.data_dict[key].index(x1)
                x2_id = self.data_dict[key].index(x2)

                # 训练集向量保存位置
                file_name = f"{self.mode}_{key}.npy"
                x1 = self.vec_data[file_name][x1_id]
                x2 = self.vec_data[file_name][x2_id]

                vectors_1.append(x1)
                vectors_2.append(x2)
                labels.append(y)
                if len(vectors_1) == self.batch_size or i == idxs[-1]:
                    labels = np.array(labels)
                    vectors_1 = np.array(vectors_1)
                    vectors_2 = np.array(vectors_2)
                    yield [vectors_1, vectors_2], labels
                    vectors_1, vectors_2, labels = [], [], []
