# -*- coding: utf-8 -*-

import os
import sys
import time
import logging


"""
用法：
import logger
LOG = logger.Logger("debug")

LOG.critical("这是一个 critical 级别的问题！")
LOG.error("这是一个 error 级别的问题！")
LOG.warning("这是一个 warning 级别的问题！")
LOG.info("这是一个 info 级别的问题！")
LOG.debug("这是一个 debug 级别的问题！")

"""


def singleton(cls):
    '''
    构建日志函数主函数
    :param cls: 日志类名称
    :return: 日志类
    '''
    instances = {}

    def getinstance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return getinstance


@singleton
class Logger:
    def __init__(self, set_level="ERROR",
                 name=os.path.split(os.path.splitext(sys.argv[0])[0])[-1],
                 log_name=time.strftime("%Y-%m-%d.log", time.localtime()),
                 log_path=None,
                 use_console=True):
        """
        :param set_level: 日志级别["NOTSET"|"DEBUG"|"INFO"|"WARNING"|"ERROR"|"CRITICAL"]
        :param name: 日志中打印的name，默认为运行程序的name
        :param log_name: 日志文件的名字，默认为当前时间（年-月-日.log）
        :param log_path: 日志文件夹的路径，默认为logger.py同级目录中的log文件夹
        :param use_console: 是否在控制台打印，默认为True
        """
        arg0 = sys.argv[0]
        if set_level is None:
            set_level = "ERROR"
        self.__logger = logging.getLogger(name)
        self.setLevel(getattr(logging, set_level.upper()) if hasattr(logging, set_level.upper()) else logging.INFO)  # 设置日志级别

        if log_path is not None:
            if not os.path.exists(log_path):  # 创建日志目录
                os.makedirs(log_path)
        formatter = logging.Formatter("[%(asctime)s %(levelname)5s] %(message)s", "%H:%M:%S")
        handler_list = list()
        if log_path is not None:
            handler_list.append(logging.FileHandler(os.path.join(log_path, log_name), encoding="utf-8"))
        if use_console:
            handler_list.append(logging.StreamHandler())
        for handler in handler_list:
            handler.setFormatter(formatter)
            self.addHandler(handler)

    def __getattr__(self, item):
        return getattr(self.logger, item)

    @property
    def logger(self):
        return self.__logger

    @logger.setter
    def logger(self, func):
        self.__logger = func


if __name__ == "__main__":

    LOG = Logger("info")

    LOG.critical("这是一个 critical 级别的问题！")
    LOG.error("这是一个 error 级别的问题！")
    LOG.warning("这是一个 warning 级别的问题！")
    LOG.info("这是一个 info 级别的问题！")
    LOG.debug("这是一个 debug 级别的问题！")

    # 只有前两个log
    LOG1 = Logger()
    LOG1.critical("LOG1 这是一个 critical 级别的问题！")
    LOG1.error("LOG1 这是一个 error 级别的问题！")
    LOG1.warning("LOG1 这是一个 warning 级别的问题！")

    LOG.info("上个LOG：这是一个 info 级别的问题！")
