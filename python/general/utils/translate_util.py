#!/usr/bin/env python
# -*- coding:utf-8 -*-

import json

import time
from config.config import Config
from urllib.error import HTTPError
from utils.http_util import http_post
from langdetect import detect
import re
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s -%(message)s")
config = Config()
IS_DEBUG = True


def debug(*msg):
    if IS_DEBUG:
        size = len(msg)
        if msg is None or size < 1:
            print("")
            return

        print("[DEBUG] ", end="")
        for s in msg:
            print(s, " ", end="")
        print("")


def translate_any_2_anyone(article, target="en"):
    """
    临时用的，将任意语言的文章翻译为英文

    :param article: String.文章
    :param target: String.可选值有en、zh等

    """
    # article = article.replace(" ", "")
    # article = filter_tags(article)
    url = config.translate_url
    article_t = ""
    try:
        article_detect = detect(article)
        if article_detect == 'ja' or article_detect == 'zh-cn':
            article_array = article.split("。")
            for sentence in article_array:
                data = {"from": "auto", "to": target, "apikey": config.translate_user_key, "src_text": sentence}
                cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                debug(f"-----------翻译句子开始----------- : {cur_time}")
                res = http_post(data, url)
                cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                debug(f"-----------翻译句子结束----------- : {cur_time}")
                res_dict = json.loads(res)
                if "tgt_text" in res_dict:
                    content = res_dict['tgt_text']
                    article_t += content + ". "
        else:
            article_array = article.split(".")
            for sentence in article_array:
                data = {"from": "auto", "to": target, "apikey": config.translate_user_key, "src_text": sentence}
                cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                debug(f"-----------翻译句子开始----------- : {cur_time}")
                res = http_post(data, url)
                cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                debug(f"-----------翻译句子结束----------- : {cur_time}")
                res_dict = json.loads(res)
                if "tgt_text" in res_dict:
                    content = res_dict['tgt_text']
                    article_t += content + ". "
        return article_t
    except HTTPError as e:
        logging.error('翻译时发生的错误：', e)
        return ''


def filter_tags(htmlstr):
    # 先过滤CDATA
    re_cdata = re.compile('//<!\[CDATA\[[^>]*//\]\]>', re.I)
    re_script = re.compile('<\s*script[^>]*>[^<]*<\s*/\s*script\s*>', re.I)
    re_style = re.compile('<\s*style[^>]*>[^<]*<\s*/\s*style\s*>', re.I)
    re_br = re.compile('<br\s*?/?>')  # 处理换行
    re_img = re.compile('< img\s*?/?>')  # 处理换行
    re_h = re.compile('</?\w+[^>]*>')  # HTML标签
    re_comment = re.compile('<!--[^>]*-->')  # HTML注释
    s = re_cdata.sub('', htmlstr)  # 去掉CDATA
    s = re_script.sub('', s)  # 去掉SCRIPT
    s = re_style.sub('', s)  # 去掉style
    s = re_br.sub('\n', s)  # 将br转换为换行
    s = re_img.sub(' ', s)  # 将br转换为换行
    s = re_h.sub('', s)  # 去掉HTML 标签
    s = re_comment.sub('', s)  # 去掉HTML注释
    # 去掉多余的空行
    blank_line = re.compile('\n+')
    s = blank_line.sub('\n', s)
    s = replaceCharEntity(s)  # 替换实体
    return s


def replaceCharEntity(htmlstr):
    CHAR_ENTITIES = {'nbsp': ' ', '160': ' ',
                     'lt': '<', '60': '<',
                     'gt': '>', '62': '>',
                     'amp': '&', '38': '&',
                     'quot': '"', '34': '"', }

    re_charEntity = re.compile(r'&#?(?P<name>w+);')
    sz = re_charEntity.search(htmlstr)
    while sz:
        entity = sz.group()  # entity全称，如>
        key = sz.group('name')  # 去除&;后entity,如>为gt
        try:
            htmlstr = re_charEntity.sub(CHAR_ENTITIES[key], htmlstr, 1)
            sz = re_charEntity.search(htmlstr)
        except KeyError:
            # 以空串代替
            htmlstr = re_charEntity.sub('', htmlstr, 1)
            sz = re_charEntity.search(htmlstr)
    return htmlstr


def repalce(s, re_exp, repl_string):
    return re_exp.sub(repl_string, s)
