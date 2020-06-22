#!/usr/bin/env python
# -*- coding:utf-8 -*-


class Config(object):

    def __init__(self):

        # self.input_event_xlsx_path = "D:\\develop\\pycharm_workspace\\petrarch2-master\\petrarch2\\data\\event_result.xlsx"
        # self.output_path = "C:\\Users\\13616\\Desktop\\result.xlsx"

        # self.xml_paths = ['D:\\develop\\pycharm_workspace\\petrarch2-master\\petrarch2\\data\\text\\wqp2.xml']

        self.http_port = 8082
        self.translate_url = "http://api.niutrans.vip/NiuTransServer/translation?"
        self.translate_user_key = "50decafed5d5ba26b2f650355436272b"
