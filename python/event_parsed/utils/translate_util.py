#!/usr/bin/env python
# -*- coding:utf-8 -*-

from utils.http_util import http_post
from urllib.error import HTTPError
from config.config import Config

import json

config = Config()


def translate_any_2_anyone(content, source="auto", target="en"):
    """
    将任意语言的文章翻译为英文。
    :param content: String.文章或句子。
    :param source: String.可选值有autp、en、zh等。
    :param target: String.可选值有en、zh等。
    :return 返回翻译后的文本。
    """
    url = config.translate_url
    data = {"from": source, "to": target, "apikey": config.translate_user_key, "src_text": content}
    try:
        res = http_post(data, url)
        res_dict = json.loads(res)

        if "tgt_text" in res_dict:
            content = res_dict['tgt_text']
        else:
            content = res

        return content
    except HTTPError as e:
        print('翻译时发生的错误：', e)
        return ''