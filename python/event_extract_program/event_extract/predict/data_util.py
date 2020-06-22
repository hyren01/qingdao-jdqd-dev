#!/usr/bin/env python
# coding: utf-8
import re
from feedwork.utils import logger as LOG

def data_process(content):
    """
    传入字符串，对字符串进行清洗，返回清洗后的字符串。
    :param content: (str)待清洗的字符串
    :return: content(str)清洗后的字符串
    :raise:TypeError 不是字符串则报错
    """
    if not isinstance(content, str):
        LOG.error("The content you want to be washed is not string!")
        raise TypeError

    if content:
        # 剔除无用的中文字符
        content = re.sub('<.*?>', '', content)
        content = re.sub('【.*?】', '', content)
        # 剔除邮箱
        content = re.sub('/^([a-z0-9_\.-]+)@([\da-z\.-]+)\.([a-z\.]{2,6})$/', '', content)
        content = re.sub('/^[a-z\d]+(\.[a-z\d]+)*@([\da-z](-[\da-z])?)+(\.{1,2}[a-z]+)+$/', '', content)
        # 剔除URL
        content = re.sub('/^<([a-z]+)([^<]+)*(?:>(.*)<\/\1>|\s+\/>)$/', '', content)
        content = re.sub('/^[http]{4}\\:\\/\\/[a-z]*(\\.[a-zA-Z]*)*(\\/([a-zA-Z]|[0-9])*)*\\s?$/','',content)
        # 剔除16进制值
        content = re.sub('/^#?([a-f0-9]{6}|[a-f0-9]{3})$/', '', content)
        # 剔除IP地址
        content = re.sub('/((2[0-4]\d|25[0-5]|[01]?\d\d?)\.){3}(2[0-4]\d|25[0-5]|[01]?\d\d?)/', '', content)
        content = re.sub('/^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$/', '',
            content)
        # 剔除用户名密码名
        content = re.sub("/^[a-z0-9_-]{3,16}$/", "", content)
        content = re.sub("/^[a-z0-9_-]{6,18}$/", "", content)
        # 剔除HTML标签
        content = re.sub('/^<([a-z]+)([^<]+)*(?:>(.*)<\/\1>|\s+\/>)$/', '', content)
        # 剔除网络字符，剔除空白符
        content = content.strip().strip('\r\n\t').replace(u'\u3000', '').replace(u'\xa0', '')
        content = content.replace('\t', '').replace(' ', '').replace('\n', '').replace('\r', '')

    return content


def get_sentences(content):
    """
    传入一篇中文文章，获取文章中的每一个句子，返回句子列表。
    :param content: (str) 一篇文章
    :return: sentences(list) 分句后的列表
    :raise: TypeError
    """
    if not isinstance(content, str):
        LOG.error("The content you want to be split is not string!")
        raise TypeError

    sentences = [f"{sentence}。" for sentence in re.split("[。！？?!]", content) if sentence]

    return sentences


if __name__ == '__main__':

    # dir_path = './article'
    #
    # file_name  = os.listdir(dir_path)
    # for file in file_name:
    #     file_path = os.path.join(dir_path, file)
    #     print('开始处理文件{}'.format(file))
    #     content = file_reader(file_path)
    #     text = process_data(content)
    #     print(text)
    while True:
        sentence = input("请输入你要处理的语句")
        if sentence != "Q":
            sentence = data_process(sentence)
            LOG.info(sentence)
        else:
            print("没有输入")
