#!/usr/bin/env python
# -*- coding:utf-8 -*-

from urllib.parse import urlencode
from urllib.request import urlopen


def http_post(data, uri):
    """
    http POST请求。
    :param data: json数据.如：{"sentence":""}。
    :param uri: String.请求目标的uri。
    :return 返回请求响应。
    """
    data = urlencode(data)
    data = data.encode()
    res = urlopen(url=uri, data=data)
    content = res.read()

    return content