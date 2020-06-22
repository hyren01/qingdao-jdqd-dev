#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Mr Fan
# @Time: 2020年05月13
import os
import numpy as np
from hrconfig import merge_predict_config
from eventmerge.predict.data_utils import read_json, save_json
from feedwork.utils import logger as LOG

# 构建参数类
CONFIG = merge_predict_config.Config()


def execute_delete(event_id):
    """
    删除模块的主控程序，读取cameo2id,然后查看事件id是否存在字典中，并进行删除。
    :return: None
    :raise: FileNotFoundError
    """
    if not os.path.exists(CONFIG.cameo2id_path) or not os.path.isfile(CONFIG.cameo2id_path):
        LOG.error(f"{CONFIG.cameo2id_path} miss, can not exec delete!")
        raise FileNotFoundError
    cameo2id = read_json(CONFIG.cameo2id_path)

    status = 0

    LOG.info("Begin to scan cameo2id dict...")
    for cameo in list(cameo2id.keys()):

        if event_id in cameo2id[cameo]:
            status += 1
            # 读取向量的文件地址
            read_file_path = os.path.join(CONFIG.vec_data_dir, f"{cameo}.npy")
            # 保存向量时的路径，只是比上边缺少了.npy,函数会自动补齐
            save_file_path = os.path.join(CONFIG.vec_data_dir, cameo)
            # 读取cameo保存的向量文件
            x = np.load(read_file_path)

            # 删除向量
            temp = np.delete(x, list(cameo2id[cameo]).index(event_id), axis=0)
            # 删除列表中的事件id
            cameo2id[cameo].remove(event_id)

            # 将更新后的文件重新保存
            # cameo对应的向量没有删除完，cameo2id[cameo]也没有删完
            if temp.shape[0] and cameo2id[cameo]:
                save_json(cameo2id, CONFIG.cameo2id_path)
                # 将向量保存到文件中
                np.save(save_file_path, temp)
                # 跳出循环
                break

            # cameo对应的向量已经置了，且cameo对应的列表也空了
            elif not temp.shape[0] and not cameo2id[cameo]:
                # 将向量文件删除
                os.remove(read_file_path)
                # 删除cameo在字典中的键
                del cameo2id[cameo]
                # 判断字典是否为空
                if len(cameo2id):
                    # 字典不为空则重新保存
                    save_json(cameo2id, CONFIG.cameo2id_path)
                else:
                    # 字典空了，则删除字典文件
                    os.remove(CONFIG.cameo2id_path)
                # 跳出循环
                break

            else:
                LOG.error("vector file does not match cameo2id, please delete file manually!")
                raise ValueError

    return status
