# coding:utf-8
# 保存向量文件
import os
import numpy as np
import json
from hrconfig import merge_predict_config
from feedwork.utils import logger as LOG

# 构建参数类
CONFIG = merge_predict_config.Config()


def save_vec_data(cameo, event_id, main_vec):
    """
    传入事件cameo、事件id、事件短句向量，将向量保存到文件中，临时保存，后期改写为保存到数据库中,
    将事件id存放到以cameo为键的字典中{cameo:[event_id], }
    :param cameo: (str)事件cameo号
    :param event_id: (str)事件编号
    :param main_vec: (ndarray)事件向量
    :return: None
    :raise:字典文件缺失/ 向量文件文件缺失 FileNotFoundError 事件id重复 ValueError 传入类型错误 TypeError
    """
    if not isinstance(cameo, str):
        LOG.error("cameo编号格式错误!")
        raise TypeError
    if not isinstance(event_id, str):
        LOG.error("事件编号格式错误!")
        raise TypeError

    # 读取向量时的路径
    read_file_path = os.path.join(CONFIG.vec_data_dir, f"{cameo}.npy")
    # 保存向量时的路径，只是比上边缺少了.npy,函数会自动补齐
    save_file_path = os.path.join(CONFIG.vec_data_dir, cameo)

    # 判断文件是否存在
    if not os.path.exists(CONFIG.cameo2id_path):
        cameo2id = {cameo: [event_id]}
        with open(CONFIG.cameo2id_path, "w", encoding="utf-8") as f:
            cameo2id = json.dumps(cameo2id, ensure_ascii=False, indent=4)
            f.write(cameo2id)
        # 将向量保存到文件中
        np.save(save_file_path, np.array([main_vec]))

    elif not os.path.exists(read_file_path):
        with open(CONFIG.cameo2id_path, "r", encoding="utf-8") as f:
            cameo2id = f.read()
        # cameo2id 字典 {cameo:[]}
        cameo2id = json.loads(cameo2id)
        # 将事件id添加到字典中
        cameo2id[cameo] = [event_id]

        # 将向量保存到文件中
        np.save(save_file_path, np.array([main_vec]))

        # 写入文件中
        with open(CONFIG.cameo2id_path, "w", encoding="utf-8") as f:
            cameo2id = json.dumps(cameo2id, ensure_ascii=False, indent=4)
            f.write(cameo2id)

    elif not os.path.exists(CONFIG.cameo2id_path):
        LOG.error("字典文件缺失!")
        raise FileNotFoundError

    else:
        with open(CONFIG.cameo2id_path, "r", encoding="utf-8") as f:
            cameo2id = f.read()
        # cameo2id 字典 {cameo:[]}
        cameo2id = json.loads(cameo2id)

        if event_id not in cameo2id.setdefault(cameo, []):
            # 将事件id保存到cameo2id字典中
            cameo2id[cameo].append(event_id)

            # 读取向量文件
            x = np.load(read_file_path)
            # 将向量拼接进去
            temp = np.vstack([x, np.array([main_vec])])
            # 将向量保存到文件中
            np.save(save_file_path, temp)

            # 写入文件中
            with open(CONFIG.cameo2id_path, "w", encoding="utf-8") as f:
                cameo2id = json.dumps(cameo2id, ensure_ascii=False, indent=4)
                f.write(cameo2id)

        else:
            LOG.error("事件id重复！")
            raise ValueError
