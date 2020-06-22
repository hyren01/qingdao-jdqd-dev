# coding: utf-8
import sys
import os

file = os.path.abspath(__file__)
dir_path = os.path.dirname(file)
sys.path.append(dir_path)

import re
import logger

LOG = logger.Logger("info")


def file_reader(file_path):
    '''
    读取路径中的文件，返回读取的字符串
    :param file_path: 文件路径
    :return: 读取到的字符串
    '''

    try:
        with open(file_path,'r') as file:
            content = file.read()
        return content
    except Exception as e:
        with open(file_path,'r',encoding='utf-8') as file:
            content = file.read()
        return content


def data_process(content):
    '''
    对传入的中文字符串进行清洗，去除邮箱、URL等无用信息。
    :param content: 待清洗的中文字符串
    :return: 清洗后的中文字符串
    '''

    if content:
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
        content = re.sub('/^[a-z0-9_-]{3,16}$/', '', content)
        content = re.sub('/^[a-z0-9_-]{6,18}$/', '', content)
        # 剔除HTML标签
        content = re.sub('/^<([a-z]+)([^<]+)*(?:>(.*)<\/\1>|\s+\/>)$/', '', content)
        # 剔除网络字符，剔除空白符
        content = content.strip().strip('\r\n\t').replace(u'\u3000', '').replace(u'\xa0', '')
        content = content.replace('\t', '').replace(' ', '').replace('\n', '').replace('\r', '')

        # 将标志性城市后插入所属国家首尔--首尔（韩国）
        city_country = {'纽约': '美国', '北京': '中国',
                        '首尔': '韩国', '平壤': '朝鲜',
                        '东京': '日本', '莫斯科': '俄罗斯',
                        '白宫': '美国', '东仓里': '朝鲜'}
        for city in city_country:
            if city in content:
                content = content.replace(city, city + '({})'.format(city_country[city]))

    return content


def process_data(data):
    '''
    判断传入的数据是否为字符串或者空，并调用清洗模块进行清洗
    :param data:传入的数据
    :return:清洗后的数据
    :raise: 如果为空或者格式错误则提示数据为空及格式错误
    '''

    if not data:
        LOG.info('The data is empty.')
        exit(1)

    elif type(data) == str:
        out_put_data = data_process(data)
        return out_put_data

    else:
        LOG.info('Sorry, the type of data you input is wrong. Please input text str.')
        exit(1)



if __name__ == '__main__':


    dir_path = './article'

    file_name  = os.listdir(dir_path)
    for file in file_name:
        file_path = os.path.join(dir_path, file)
        print('开始处理文件{}'.format(file))
        content = file_reader(file_path)
        text = process_data(content)
        print(text)

