# -*- coding: utf-8 -*-


class Config(object):

    def __init__(self):
        # JDBC 输入源配置
        self.input_jdbc_schema1 = "jdqddb"           # 管理数据所在库
        self.input_jdbc_user1 = "jdqd"
        self.input_jdbc_password1 = "jdqd"
        self.input_jdbc_host1 = "139.9.126.19"
        self.input_jdbc_port1 = "31001"
        self.input_jdbc_schema2 = "jdqddb2"          # 模型使用数据所在库
        self.input_jdbc_user2 = "jdqd"
        self.input_jdbc_password2 = "jdqd"
        self.input_jdbc_host2 = "139.9.126.19"
        self.input_jdbc_port2 = "31001"
        # 模型存放目录
        self.model_root_path = "C:/Users/13616/Desktop/events_predict/"
        # 预测时临时数据存放目录
        self.data_root_path = "C:/Users/13616/Desktop/events_predict/"
