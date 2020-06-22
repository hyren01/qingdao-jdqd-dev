#!/usr/bin/env python
# -*- coding:utf-8 -*-

from config.config import Config

config = Config()
global_event_tag = config.global_event_tag


def __node2json(node):
    """
    将图数据库中查询出来的事件节点对象转换为json对象。

    :param node: object.事件节点
    :return json数据。
    """
    json_result = {}
    for key in node.keys():
        json_result[key] = node[key]
    return json_result


def __create_events(graph_db, events, event_relations):
    """
    在图数据库中创建事件节点和关系节点，若事件节点或关系节点传入空数组，则不会根据空数组创建节点。

    :param graph_db: object.图数据库连接对象
    :param events: array.事件节点数据，如：[{"event_name":"", "event_tag":"", "event_attribute":{event_id:'',
                                            event_sentence:'', event_date:''}}]
    :param event_relations: array.关系节点数据，如：[{"source_event_tag":"", "target_event_tag":"",
                                            "source_event_id":"", "target_event_id":"", "realtion":'',
                                            "relation_attribute":{name:''}}]
    :return 事件节点和关系节点是否创建成功。
    """
    tx = graph_db.begin_transaction()
    try:
        for event in events:
            if 'event_tag' in event:
                event_tag = event["event_tag"]
            else:
                event_tag = global_event_tag
            tx.run(
                "CREATE (:{event_tag} {event_attribute})".format(event_tag=event_tag,
                                                                 event_attribute=str(event["event_attribute"]))
            )
        for relation in event_relations:
            if 'source_event_tag' in relation:
                source_event_tag = relation["source_event_tag"]
            else:
                source_event_tag = global_event_tag
            if 'target_event_tag' in relation:
                target_event_tag = relation["target_event_tag"]
            else:
                target_event_tag = global_event_tag
            tx.run(
                ("MATCH (event1:{source_event_tag}),(event2:{target_event_tag}) WHERE "
                 "event1.event_id = '{source_event_id}' AND event2.event_id = '{target_event_id}' "
                 "CREATE (event1)-[:{relation} {relation_attribute}]->(event2)").format(
                    source_event_tag=source_event_tag, target_event_tag=target_event_tag,
                    source_event_id=relation["source_event_id"], target_event_id=relation["target_event_id"],
                    relation=relation["relation"], relation_attribute=relation["relation_attribute"]
                )
            )
        tx.commit()
        return True
    except RuntimeError:
        tx.rollback()
        return False
#
#
# def __create_events(graph_db, events, event_relations):
#     """
#     在图数据库中创建事件节点和关系节点，若事件节点或关系节点传入空数组，则不会根据空数组创建节点。
#
#     :param graph_db: object.图数据库连接对象
#     :param events: array.事件节点数据，如：[{"event_name":"", "event_tag":"", "event_attribute":{event_id:'',
#                                             event_sentence:'', event_date:''}}]
#     :param event_relations: array.关系节点数据，如：[{"source_event_tag":"", "target_event_tag":"",
#                                             "source_event_id":"", "target_event_id":"", "realtion":'',
#                                             "relation_attribute":{name:''}}]
#     :return 事件节点和关系节点是否创建成功。
#     """
#     tx = graph_db.begin_transaction()
#     try:
#         for event in events:
#             if 'event_tag' in event:
#                 event_tag = event["event_tag"]
#             else:
#                 event_tag = global_event_tag
#             tx.run(
#                 "CREATE (:{event_tag} {event_attribute})".format(event_tag=event_tag,
#                                                                  event_attribute=str(event["event_attribute"]))
#             )
#         for relation in event_relations:
#             if 'source_event_tag' in relation:
#                 source_event_tag = relation["source_event_tag"]
#             else:
#                 source_event_tag = global_event_tag
#             if 'target_event_tag' in relation:
#                 target_event_tag = relation["target_event_tag"]
#             else:
#                 target_event_tag = global_event_tag
#             tx.run(
#                 ("MATCH (event1:{source_event_tag}),(event2:{target_event_tag}) WHERE "
#                  "event1.event_id = '{source_event_id}' AND event2.event_id = '{target_event_id}' "
#                  "CREATE (event1)-[:{relation} {relation_attribute}]->(event2)").format(
#                     source_event_tag=source_event_tag, target_event_tag=target_event_tag,
#                     source_event_id=relation["source_event_id"], target_event_id=relation["target_event_id"],
#                     relation=relation["relation"], relation_attribute=relation["relation_attribute"]
#                 )
#             )
#         tx.commit()
#         return True
#     except RuntimeError:
#         tx.rollback()
#         return False


def create_event(graph_db, event_id, short_sentence, event_datetime, event_tag):
    """
    在图数据库中创建事件节点。

    :param graph_db: object.图数据库连接对象
    :param event_id: string.事件id
    :param short_sentence: string.事件短句
    :param event_datetime: string.事件发生日期
    :param event_tag: 事件节点标签（默认Event）
    :return 事件节点是否创建成功。
    """
    if event_tag is None:
        event_tag = global_event_tag
    event_attribute = "event_id: '{event_id}', event_sentence: '{short_sentence}', event_date: " \
                      "'{event_datetime}'".format(event_id=event_id, short_sentence=short_sentence,
                                                  event_datetime=event_datetime)
    event_attribute = "{" + event_attribute + "}"
    event = [{"event_name": "", "event_tag": event_tag, "event_attribute": event_attribute}]
    success = __create_events(graph_db, event, [])

    return success


def create_relation(graph_db, cause_event_id, effect_event_id, relation_type, event_tag):
    """
    在图数据库中创建关系节点。

    :param graph_db: object.图数据库连接对象
    :param cause_event_id: string.源节点事件id
    :param effect_event_id: string.目前节点事件id
    :param relation_type: string.关系类型
    :param event_tag: 事件节点标签（默认Event）
    :return 关系节点是否创建成功。
    """
    if event_tag is None:
        event_tag = global_event_tag
    relation_attribute = "name: '{name}'".format(name=relation_type)
    relation_attribute = "{" + relation_attribute + "}"
    relation = [{"source_event_tag": event_tag, "target_event_tag": event_tag,
                 "source_event_id": cause_event_id, "target_event_id": effect_event_id,
                 "relation": relation_type, "relation_attribute": relation_attribute}]
    success = __create_events(graph_db, [], relation)
    return success
    # if success != "success":
    #     print("创建关系节点失败")


def get_event_by_ids(graph_db, event_ids, event_tag, start_date, end_date):
    """
    在图数据库中根据事件id数组查询出对应的事件节点。

    :param graph_db: object.图数据库连接对象
    :param event_ids: array.事件id数组，如：['','']
    :param event_tag: string.事件标签（默认Event）
    :param start_date: string.事件发生开始日期
    :param end_date: string.事件发生结束日期
    :return 事件节点数组。
    """
    if event_tag is None:
        event_tag = global_event_tag
    cql_str = "MATCH (event:{event_tag}) WHERE event.event_id IN {ids}".format(event_tag=event_tag, ids=event_ids)
    if start_date is not None and start_date != '' and start_date != 'None':
        cql_str = cql_str + " AND event.event_date > {start_date}".format(start_date=start_date)
    if end_date is not None and end_date != '' and end_date != 'None':
        if start_date is not None and start_date != '' and start_date != 'None':
            cql_str = cql_str + " OR "
        else:
            cql_str = cql_str + " AND "
        cql_str = cql_str + "event.event_date < {end_date}".format(end_date=start_date)
    cql_str = cql_str + " RETURN event"
    results = graph_db.run(cql_str)
    event_result = []
    for record in results:
        event_result.append(__node2json(record["event"]))

    return event_result


def get_event_rel_by_id(graph_db, event_id, event_tag):
    """
    在图数据库中根据事件id查询出对应的跟该事件相关联的所有事件节点及其对应关系。

    :param graph_db: object.图数据库连接对象
    :param event_id: string.事件id
    :param event_tag: string.事件标签（默认Event）

    :return 事件节点及关系数组，如：[{"source_event_id":"", "source_event_sentence":"", "source_event_date":"",
                                      "relation_name":"", "target_event_id":"", "target_event_sentence":"",
                                      "target_event_date":""}]。
    """
    if event_tag is None:
        event_tag = global_event_tag
    cql_str = "MATCH (event1:{event_tag})-[relation]->(event2:{event_tag}) WHERE event1.event_id = '{event_id}' OR " \
              "event2.event_id = '{event_id}' RETURN event1,relation,event2".format(event_tag=event_tag,
                                                                                    event_id=event_id)
    results = graph_db.run(cql_str)
    result = []
    for record in results:
        source_event = __node2json(record["event1"])
        event_relation = __node2json(record["relation"])
        target_event = __node2json(record["event2"])
        result.append({'source_event_id': source_event['event_id'],
                       'source_event_sentence': source_event['event_sentence'],
                       'source_event_date': source_event['event_date'],
                       'relation_name': event_relation['name'],
                       'target_event_id': target_event['event_id'],
                       'target_event_sentence': target_event['event_sentence'],
                       'target_event_date': target_event['event_date']})

    return result


def get_event_to_event(graph_db, event_source_sentence, event_target_sentence):
    cql_str = f"MATCH p=(event1:Event)-[relation]->(event2:Event) where event1.event_sentence='{event_source_sentence}' or event2.event_sentence='{event_target_sentence}' RETURN event1,relation,event2"
    results = graph_db.run(cql_str)
    result = []
    result_d = []
    wstr = ""
    for record in results:
        source_event = __node2json(record["event1"])
        event_relation = __node2json(record["relation"])
        target_event = __node2json(record["event2"])
        result_d.append(source_event['event_id'])
        result_d.append(target_event['event_id'])
        result.append({'source_event_id': source_event['event_id'],
                       'source_event_sentence': source_event['event_sentence'],
                       'source_event_date': source_event['event_date'],
                       'relation_name': event_relation['name'],
                       'target_event_id': target_event['event_id'],
                       'target_event_sentence': target_event['event_sentence'],
                       'target_event_date': target_event['event_date']})
    cql_str = f"MATCH p=(event1:Event)-[relation]->(event2:Event) where event1.event_sentence='{event_target_sentence}' or event2.event_sentence='{event_source_sentence}' RETURN event1,relation,event2"
    results = graph_db.run(cql_str)
    for record in results:
        source_event = __node2json(record["event1"])
        event_relation = __node2json(record["relation"])
        target_event = __node2json(record["event2"])
        result_d.append(source_event['event_id'])
        result_d.append(target_event['event_id'])
        result.append({'source_event_id': source_event['event_id'],
                       'source_event_sentence': source_event['event_sentence'],
                       'source_event_date': source_event['event_date'],
                       'relation_name': event_relation['name'],
                       'target_event_id': target_event['event_id'],
                       'target_event_sentence': target_event['event_sentence'],
                       'target_event_date': target_event['event_date']})
    for i in range(len(result_d)):
        if i > 0:
            wstr += " or "
        wstr += "event1.event_id='" + result_d[i] + "' "
    print(wstr)
    cql_str = "MATCH (event1:Event) where " + wstr + "return event1"
    results_data = graph_db.run(cql_str)
    result_d = []
    for record in results_data:
        event = __node2json(record["event1"])
        result_d.append({'event_id': event['event_id'],
                         'event_sentence': event['event_sentence']})
    print(result_d)
    return True, result,result_d
