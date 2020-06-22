#!/usr/bin/env python
# coding : utf-8
from urllib.parse import urlencode
from urllib.request import urlopen
from urllib.error import HTTPError
import json
from hrconfig import event_extract_predict_config
from feedwork.utils import logger as LOG

# 参数对象
CONFIG = event_extract_predict_config.Config()


def transform_any_2_en(article):
    """
    将任意语言的文章翻译为英文
    :param article: 文章--str
    :return: 英文文章--str
    :raise:TypeError
    """
    if not isinstance(article, str):
        LOG.error("待翻译为英文的内容格式错误，需要字符串格式！")
        raise TypeError

    data = {"from": "auto", "to": "en", "apikey": CONFIG.user_key, "src_text": article}
    try:
        data_en = urlencode(data)
        req = f"{CONFIG.translate_url}&{data_en}"
        res = urlopen(req)
        res = res.read()
        res_dict = json.loads(res)

        if "tgt_text" in res_dict:
            content = res_dict['tgt_text']
        else:
            content = res

        return content
    except HTTPError:
        LOG.info('翻译时发生的错误，通常是http请求太大')
        LOG.info(str(HTTPError))
        return ''


def transform_any_2_zh(article):
    """
    将任意语言的文章翻译为中文
    :param article: 文章--str
    :return: 中文文章--str
    :raise:TypeError
    """
    if not isinstance(article, str):
        LOG.error("待翻译为中文的内容格式错误，需要字符串格式！")
        raise TypeError
    data = {"from": "auto", "to": "zh", "apikey": CONFIG.user_key, "src_text": article}
    try:
        data_en = urlencode(data)
        req = f"{CONFIG.translate_url}&{data_en}"
        res = urlopen(req)
        res = res.read()
        res_dict = json.loads(res)

        if "tgt_text" in res_dict:
            content = res_dict['tgt_text']
        else:
            content = res

        return content
    except HTTPError:
        LOG.info('翻译时发生的错误，通常是http请求太大')
        LOG.info(str(HTTPError))
        return ''


if __name__ == "__main__":

    article = "小明给在山区支教的同学打了一个电话。"
    trans = transform_any_2_en(article)
