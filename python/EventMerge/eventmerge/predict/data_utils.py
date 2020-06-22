#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Mr Fan
# @Time: 2020年05月13
import os
import json
from feedwork.utils import logger as LOG



def read_json(file_path):
    """
    传入json文件路径返回解析后的文件内容。
    :param file_path: (str)json文件路径
    :return: data解析后的文件内容
    """
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        LOG.error(f"{file_path} miss!")
        raise FileNotFoundError

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
    except TypeError:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()

    data = json.loads(content)

    return data


def save_json(data, file_path):
    """
    传入数据，使用json解析后保存到json文件中。
    :param data: 需要保存到json文件中的数据。
    :param file_path: (str)保存json文件的地址
    :return: None
    """
    data = json.dumps(data, ensure_ascii=False, indent=4)

    with open(file_path, "w", encoding="utf-8") as file:
        file.write(data)
