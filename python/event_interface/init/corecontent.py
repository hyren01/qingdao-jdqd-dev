#!/usr/bin/env python
# -*- coding:utf-8 -*-
import json
import psycopg2
import psycopg2.extras
from urllib.parse import urlencode
from urllib.request import urlopen


def http_post(data, uri):

    data = urlencode(data)
    data = data.encode()
    res = urlopen(url=uri, data=data)
    content = res.read()

    return content


def init_t_article_msg_zh():
    postgres_db = psycopg2.connect(host="139.9.126.19", port="31001", database="ebmdb2", user="jdqd", password="jdqd")
    cursor = postgres_db.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cursor.execute("SELECT article_id, content FROM t_article_msg_en")
    result = cursor.fetchall()
    result = json.loads(json.dumps(result))
    i = 1
    for row in result:
        print(str(i))
        content = row["content"]
        if content is None:
            continue
        data = {"content": content}
        res = http_post(data, "http://172.168.0.115:38082/coref_with_content")
        res_dict = json.loads(res)
        if res_dict['status'] != 'success':
            print("在进行指代消解时发生异常", res_dict)
            continue
        content = res_dict["coref"]     # 指代消解后返回的是中文
        cursor.execute("INSERT INTO t_article_msg_zh(article_id, content) VALUES (%s,%s)", [row["article_id"], content])
        i = i + 1
    postgres_db.commit()
    postgres_db.close()
