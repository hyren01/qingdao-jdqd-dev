#!/usr/bin/env python
# -*- coding:utf-8 -*-

import json
import psycopg2
import psycopg2.extras

from config.config import Config
from utils.translate_util import translate_any_2_anyone
from utils.http_util import http_post

config = Config()


def init_t_article_msg_en():
    postgres_db = psycopg2.connect(host=config.db_host, port=config.db_port, database=config.db_name,
                                   user=config.db_user, password=config.db_passwd)
    cursor = postgres_db.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cursor.execute("SELECT article_id, content_cleared FROM t_article_msg WHERE "
                   "substring(article_id, 1, 1) IN ('0')")
    result = cursor.fetchall()
    result = json.loads(json.dumps(result))
    i = 1
    for row in result:
        print(str(i))
        content = row["content_cleared"]
        if content is None:
            continue
        content = translate_any_2_anyone(content, target="en")
        cursor.execute("INSERT INTO t_article_msg_en(article_id, content) VALUES (%s,%s)", [row["article_id"], content])
        i = i + 1
    postgres_db.commit()
    postgres_db.close()


def init_t_article_msg_zh():
    postgres_db = psycopg2.connect(host=config.db_host, port=config.db_port, database=config.db_name,
                                   user=config.db_user, password=config.db_passwd)
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
        res = http_post(data, config.coref_interface_uri)
        res_dict = json.loads(res)
        if res_dict['status'] != 'success':
            print("在进行指代消解时发生异常", res_dict)
            continue
        content = res_dict["coref"]     # 指代消解后返回的是中文
        cursor.execute("INSERT INTO t_article_msg_zh(article_id, content) VALUES (%s,%s)", [row["article_id"], content])
        i = i + 1
    postgres_db.commit()
    postgres_db.close()


if __name__ == '__main__':
    # init_t_article_msg_en()
    init_t_article_msg_zh()
