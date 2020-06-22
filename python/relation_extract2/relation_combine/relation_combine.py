import requests
import json
from itertools import product
from loguru import logger

# 事件抽取
url_event_extract = "http://172.168.0.115:38082/event_extract"
# 提取关键词
url_relation_keywords = "http://172.168.0.115:12319/relation_keywords"
# 拆分子句
url_relation_split = "http://172.168.0.115:12320/relation_split"
# 事件关系预测
url_relation_classify = "http://172.168.0.115:12318/relation_classify"


def request(url, data):
    req = requests.post(url, data)
    return json.loads(req.text)


# def exe(sentence):
# chars = [' ', '\t', u'\u3000', '\n']
id = '999'
sentence = '尽管恢复了许多旧苏联时期的警察国家模式，并且不断有“我们正在与北约开战”的宣传，但越来越多的俄罗斯人显然在指责弗拉基米尔普京和弗拉基米尔普京的政策导致了生活水平的下降，失业率的上升以及未来更多的坏事'


def concat_svo(events):
    svos = []
    for e in events:
        verb = e.get('verb')
        if not verb:
            continue
        subject = e.get('subject')
        object = e.get('object')
        svo = ''.join([subject, verb, object])
        svos.append(svo)
    return svos


def extract(sentence):
    sentence = sentence.replace(' ', '').replace('\t', '').replace('\n', '').replace(u'\u3000', u'')
    # 1、调用提取关键词接口
    req_keywords_data = {'sentence': sentence}
    keywords = request(url_relation_keywords, req_keywords_data)
    logger.info('keywords,', keywords)
    # 单个关键词，如“导致”
    single = keywords['single']
    # 多个关键词，如“因为-所以”
    multi1 = keywords['multi1']
    multi2 = keywords['multi2']
    if single:
        keywords = [[w] for w in single]
        # keywords = [single]
    elif multi1 and multi2:
        # 如果两个值有一个为空，keywords就为空
        keywords = list(product(multi1, multi2))
    else:
        return keywords, [], [], [], 1

    splits = []

    for k in keywords:
        # 调用拆分子句
        req_split_data = {'sentence': sentence, 'keyword': json.dumps(k, ensure_ascii=False)}
        split = request(url_relation_split, req_split_data)
        if not split:
            continue
        splits.extend(split)

    if not splits:
        return keywords, [], [], [], 2

    # 左句事件短语
    left_event = []
    # 右句事件短语
    right_event = []
    for s in splits:
        # 左句
        left_sentence = s[0]
        # 右句
        right_sentence = s[1]
        # 调用事件抽取
        req_left_data = {'sentence': left_sentence}
        req_right_data = {'sentence': right_sentence}
        left_resp = request(url_event_extract, req_left_data)
        # 左句右句调用事件抽取如果其中一个为空，则跳过
        if not left_resp:
            continue
        right_resp = request(url_event_extract, req_right_data)
        if not right_resp:
            continue
        left_data = left_resp.get('data')
        if left_data is None:
            continue
        right_data = right_resp.get('data')
        if right_data is None:
            continue
        left_events = left_data[0]['events']
        right_events = right_data[0]['events']
        left_event.extend(concat_svo(left_events))
        right_event.extend(concat_svo(right_events))
    logger.info('svos,', left_event, right_event)

    # 组成需要进行关系匹配的事件短语
    event_pairs = list(product(left_event, right_event))
    if not event_pairs:
        return keywords, splits, [left_event, right_event], [], 3
    logger.info('event_pairs,', event_pairs)

    rst = []

    for p in event_pairs:
        # 调用关系抽取
        req_classify_data = {'event1': p[0], 'event2': p[1]}
        classify_resq = request(url_relation_classify, req_classify_data)
        rst.append({'event_pair': p, 'relation': classify_resq['type']})

    return keywords, splits, event_pairs, rst, 4


IS_DEBUG = True


def debug(*msg):
    if IS_DEBUG:
        size = len(msg)
        if msg is None or size < 1:
            return

        print("[DEBUG] ", end="")
        for s in msg:
            print(s, " ", end="")


if __name__ == '__main__':

    database = "ebmdb2"
    user = "jdqd"
    password = "jdqd"
    host = "139.9.126.19"
    port = "31001"

    import db_util
    import file_util
    import time

    t = time.time()
    conn = db_util.get_conn(database, user, password, host, port)
    debug(conn)
    sql = 'select sentence_id, event_sentence from ebm_event_sentence_copy1'
    rst = db_util.query(conn, sql)
    t2 = time.time()
    logger.info(f'finished query, used {t2 - t} secs')

    # f = file_util.openfile('record.txt')

    sql_insert = []

    cnt = 0
    total = len(rst)
    for s in rst:
        cnt += 1
        logger.info(cnt, 'out of', total)
        sid, st = s
        keywords, splits, event_pairs, rst, code = extract(st)
        try:
            if code != 1 and code != 4:
                s = f"insert into event_relations(sentence_id,relation_type , words, left_sentence, event_source, event_target)" \
                    f"values('{sid}','{code}', '{json.dumps(keywords, ensure_ascii=False)}', '{json.dumps(splits, ensure_ascii=False)}', '{json.dumps(event_pairs, ensure_ascii=False)}', '{json.dumps(rst, ensure_ascii=False)}')"
                # sql_insert.append(s)
                conn = db_util.get_conn(database, user, password, host, port)
                db_util.modify(conn, s)
            elif code == 4:
                for i in range(len(rst)):
                    s = f"insert into event_relations(sentence_id,relation_type , words, left_sentence, event_source, event_target,relation)" \
                        f"values('{sid}','{code}', '{json.dumps(keywords, ensure_ascii=False)}', '{json.dumps(splits, ensure_ascii=False)}', '{rst[i]['event_pair'][0]}', '{rst[i]['event_pair'][1]}','{rst[i]['relation']}')"
                    # sql_insert.append(s)
                    conn = db_util.get_conn(database, user, password, host, port)
                    db_util.modify(conn, s)
        except Exception as ex:
            conn.rollback()
            debug(f"{sid},ex.msg={ex}")

    # f.close()
