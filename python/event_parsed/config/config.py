#!/usr/bin/env python
# -*- coding:utf-8 -*-


class Config(object):

    def __init__(self):
        # self.stanfordnlp_path = "D:\\develop\\pycharm_workspace\\stanfordnlp"
        self.allennlp_constituency_parser_model_path = "D:\\develop\\pycharm_workspace\\allennlp\\elmo-constituency-parser-2018.03.14.tar.gz"
        self.allennlp_dependency_parser_model_path = "D:\\develop\\pycharm_workspace\\allennlp\\biaffine-dependency-parser-ptb-2018.08.23.tar.gz"
        self.allennlp_openie_model_path = "D:\\develop\\pycharm_workspace\\allennlp\\openie-model.2018-08-20.tar.gz"
        self.allennlp_namedentity_model_path = "D:\\develop\\pycharm_workspace\\allennlp\\ner-model-2018.12.18.tar.gz"
        self.allennlp_coref_model_path = "D:\\develop\\pycharm_workspace\\allennlp\\coref-model-2018.02.05.tar.gz"
        self.sentiment_analysis_model_path = "D:\\develop\\pycharm_workspace\\allennlp\\sst-roberta-large-2020.02.17.tar.gz"
        # self.article_dir = "D:\\develop\\pycharm_workspace\\event_extractor\\resources\\all_file"
        # self.xml_paths = 'D:\\develop\\pycharm_workspace\\petrarch2-master\\petrarch2\\data\\text\\wqp2.xml'

        self.http_port = 8081
        self.translate_url = "http://api.niutrans.vip/NiuTransServer/translation?"
        self.translate_user_key = "50decafed5d5ba26b2f650355436272b"
