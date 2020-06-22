#!/usr/bin/env python
# -*- coding:utf-8 -*-

from neo4j import GraphDatabase, basic_auth
from config.config import Config
from utils.http_util import http_post
import pandas as pd
import services.graph_database as graph_service
import services.solr_operator as solr_service
import solr
import json

config = Config()
global_event_tag = config.global_event_tag


def __get_cameo_code(sentence):
    data = {"sentence": sentence}
    res = http_post(data, config.event_extract_uri)
    response = json.loads(res)
    sentence_parsed_array = response["data"]
    sentence_parsed = sentence_parsed_array[0]
    cameo = sentence_parsed["events"][0]["cameo"]

    return cameo


def init_event_relation():
    init_xlsx_path = "关系事件20200421.xlsx"
    df = pd.read_excel(init_xlsx_path, header=0)
    gdb_driver = GraphDatabase.driver(config.neo4j_uri, auth=basic_auth(config.neo4j_username, config.neo4j_password))
    neo4j_db = gdb_driver.session()
    solr_db = solr.Solr(config.solr_uri)

    ids = []
    for index, row in df.iterrows():
        sid = row['sid']
        tid = row['tid']
        if ids.__contains__(sid) is not True:
            source_event = row['source_event']
            source_event_cameo = __get_cameo_code(source_event)
            solr_service.add_document(solr_db, sid, source_event, '', source_event_cameo)
            graph_service.create_event(neo4j_db, sid, source_event, '', None)
            ids.append(sid)

        if ids.__contains__(tid) is not True:
            target_event = row['target_event']
            target_event_cameo = __get_cameo_code(target_event)
            solr_service.add_document(solr_db, tid, target_event, '', target_event_cameo)
            graph_service.create_event(neo4j_db, tid, target_event, '', None)
            ids.append(tid)

        success = graph_service.create_relation(neo4j_db, sid, tid, row['relation'], global_event_tag)
        print("关系创建：" + str(success))

    neo4j_db.close()
    solr_db.close()


def del_event_relation():
    init_xlsx_path = "关系事件20200421.xlsx"
    df = pd.read_excel(init_xlsx_path, header=0)
    gdb_driver = GraphDatabase.driver(config.neo4j_uri, auth=basic_auth(config.neo4j_username, config.neo4j_password))
    neo4j_db = gdb_driver.session()
    solr_db = solr.Solr(config.solr_uri)
    ids = []
    for index, row in df.iterrows():
        sid = row['sid']
        tid = row['tid']
        if ids.__contains__(sid) is not True:
            ids.append(sid)
        if ids.__contains__(tid) is not True:
            ids.append(tid)

    for event_id in ids:
        solr_db.delete_query("graph-id:" + event_id)
        neo4j_db.run("MATCH (event:Event) WHERE event.event_id = '{event_id}' DETACH DELETE event".
                     format(event_id=event_id))

    neo4j_db.close()
    solr_db.close()


if __name__ == '__main__':
    init_event_relation()
