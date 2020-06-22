#!/usr/bin/env python
# -*- coding:utf-8 -*-


class Config(object):

    def __init__(self):

        self.http_port = 8083

        self.neo4j_uri = "bolt://172.168.0.115:7687"
        self.neo4j_username = "neo4j"
        self.neo4j_password = "q1w2e3"

        self.db_host = "139.9.126.19"
        self.db_port = "31001"
        self.db_name = "ebmdb"
        self.db_user = "jdqd"
        self.db_passwd = "jdqd"

        self.global_event_tag = "Event"

        self.solr_uri = 'http://172.17.0.1:8983/solr/solr_graph'
        self.fulltext_topN = 10

        self.translate_url = "http://api.niutrans.vip/NiuTransServer/translation"
        #self.translate_user_key = "dfc242105d925bf99d82bc16c2171a00"
        self.translate_user_key = "b3d33c84a6291b89524e1a759064032a"
        # 指代消解接口
        self.coref_interface_uri = "http://172.168.0.115:38082/coref_with_content"
        # 关系抽取接口
        self.relextract_interface_uri = "http://192.168.3.78:12316/relation_extract"
        # 事件抽取接口
        self.event_extract_uri = "http://172.168.0.115:38082/event_extract"
        # self.sentence_parsed_uri = "http://192.168.3.115:38081/ematch"
        # 组成成份分析接口
        self.constituency_parsed_uri = "http://127.0.0.1:8081/constituency_parsed"
        # 事件类型（CAMEO CODE）分析接口
        self.event_code_parsed_uri = "http://127.0.0.1:8082/get_event_code"
        # 文本相似度匹配接口
        self.fulltext_match_uri = "http://127.0.0.1:8083/solr_search"
        # “图数据查询接口”，在图数据库中根据事件id数组查询出对应的事件节点数组；
        self.get_events_uri = "http://127.0.0.1:8083/get_events_by_ids"
