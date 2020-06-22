#!/usr/bin/env python
# coding:utf-8
# 读取向量文件
import os
import numpy as np
import json
from hrconfig import merge_predict_config
from feedwork.utils import logger as LOG

# 构建参数类
CONFIG = merge_predict_config.Config()


def load_vec_data(cameo):
    """
    传入事件cameo号，到cameo号对应的列表中加载所有事件短句向量
    :param cameo:(str)事件cameo号
    :return:data(dict){事件id:向量}
    :raise:TypeError FileNotFoundError
    """
    read_file_path = os.path.join(CONFIG.vec_data_dir, f"{cameo}.npy")

    data = {}
    # 判断文件是否存在
    if not os.path.exists(read_file_path):
        return data

    elif not os.path.exists(CONFIG.cameo2id_path):
        LOG.error("字典文件缺失!")
        raise FileNotFoundError

    else:
        with open(CONFIG.cameo2id_path, "r", encoding="utf-8") as f:
            cameo2id = f.read()
        # cameo2id 字典 {cameo:[]}
        cameo2id = json.loads(cameo2id)

        # 读取文件中的向量
        x = np.load(read_file_path)

        LOG.info(f"开始加载向量数据{read_file_path}。。。")
        for key, value in zip(cameo2id[cameo], x):
            data[key] = value
        LOG.info(f"{read_file_path}向量数据加载完成！")

        return data
