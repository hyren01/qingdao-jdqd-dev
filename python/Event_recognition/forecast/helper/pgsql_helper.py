# -*- coding: utf-8 -*-

import psycopg2
import utils.gener_id as gi
import logging

from config import Config

conf = Config()
schema1 = conf.input_jdbc_schema1   # 管理数据所在库
user1 = conf.input_jdbc_user1
password1 = conf.input_jdbc_password1
host1 = conf.input_jdbc_host1
port1 = conf.input_jdbc_port1
schema2 = conf.input_jdbc_schema2   # 模型使用数据所在库
user2 = conf.input_jdbc_user2
password2 = conf.input_jdbc_password2
host2 = conf.input_jdbc_host2
port2 = conf.input_jdbc_port2

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s -%(message)s")


def get_pgsql_connect(jdbc_schema, jdbc_user, jdbc_password, jdbc_host, jdbc_port):

    return psycopg2.connect(database=jdbc_schema, user=jdbc_user, password=jdbc_password, host=jdbc_host, port=jdbc_port)


def get_predict_rows_by_table(table_name, need_date=False):

    # if need_date:
    #     sql = """SELECT * FROM {} ORDER BY rqsj ASC""".format(table_name)
    # else:
    #     sql = """SELECT * FROM {} ORDER BY rqsj ASC""".format(table_name)
    sql = """SELECT * FROM {} ORDER BY rqsj ASC""".format(table_name)

    conn = get_pgsql_connect(schema2, user2, password2, host2, port2)
    cursor = conn.cursor()
    cursor.execute(sql)
    data_result = cursor.fetchall()
    conn.close()

    return data_result


def get_event_rows(event_type="2"):
    event_type = "leibie" if str(event_type) == "1" else "mingcheng"
    """
        查询事件表的数据。

        :return: 返回两个对象，第一个对象为去重后的数据（获得事件日期及事件编号），形式为[[]]，第二对象为被去重的数据，
                 形式为[{"rqsj": "", "event_id": ""}]，
    """
    sql = "SELECT qssj, {} FROM dashijian WHERE qssj IS NOT NULL ORDER BY qssj ASC".format(event_type)

    conn = get_pgsql_connect(schema2, user2, password2, host2, port2)
    cursor = conn.cursor()
    cursor.execute(sql)
    # 用于记录去重后的记录
    normal_rows = []
    # 用于记录重复的第二条开始的记录
    duplicate_rows = []
    # sql通过日期升序排序查出数据，并没有去除重复，所以使用本地变量date_data来完成去重的操作，
    # 具体的去重为：记录上一个日期，若与本行数据日期相同则认为是重复数据，记录重复数据后返回以便后续使用
    date_data = None
    while True:
        row = cursor.fetchone()
        if not row:
            break
        date_data_tmp = row[0]
        if date_data_tmp != date_data:
            date_data = date_data_tmp
            normal_rows.append(tuple([date_data_tmp, row[1]]))
        elif date_data is not None:
            duplicate_rows.append({"rqsj": date_data_tmp, "event_id": row[1]})
        else:
            print("程序在处理重复数据时发生异常：" + date_data)
    conn.close()

    return normal_rows, duplicate_rows


def get_event_rows_by_duplicate_data(rqsj, event_id, event_type="2"):
    event_type = "leibie" if str(event_type) == "1" else "mingcheng"
    """
        根据指定的事件日期及事件编号查询出数据，查询出的数据中，将不为指定的日期及事件编号的数据中的事件编号置为0（含义为不是事件）

        :param rqsj: 事件日期
        :param event_id: 事件编号
        :return: 返回除了指定的日期及事件编号数据没发生改变，其它数据的事件编号置为0的全量数量
    """
    # 因为使用SQL UNION ALL会很方便的加入当前的这一条重复数据，也很方便的进行排序。
    sql = "SELECT t.rqsj, t.event_id FROM (SELECT qssj AS rqsj, '0' AS event_id FROM dashijian " \
          "WHERE qssj != %(rqsj)s AND {} != %(event_id)s UNION ALL " \
          "SELECT qssj AS rqsj, {} AS event_id FROM dashijian WHERE qssj = %(rqsj)s " \
          "AND {} = %(event_id)s) t WHERE t.rqsj IS NOT NULL ORDER BY t.rqsj ASC".\
        format(event_type, event_type, event_type)\

    conn = get_pgsql_connect(schema2, user2, password2, host2, port2)
    cursor = conn.cursor()
    cursor.execute(sql, {'rqsj': rqsj, 'event_id': event_id})
    data_result = cursor.fetchall()
    conn.close()

    return data_result


def update_model_status(model_id, status="3"):

    sql = "UPDATE t_event_model SET status = %(status)s WHERE model_id = %(model_id)s"
    conn = get_pgsql_connect(schema1, user1, password1, host1, port1)
    cursor = conn.cursor()
    cursor.execute(sql, {'status': status, 'model_id': model_id})
    conn.commit()
    conn.close()


def model_train_done(model_id, file_path_list, status="3"):

    conn = get_pgsql_connect(schema1, user1, password1, host1, port1)
    try:
        cursor = conn.cursor()
        sql = "UPDATE t_event_model SET status = %(status)s WHERE model_id = %(model_id)s"
        cursor.execute(sql, {'status': status, 'model_id': model_id})

        for element in file_path_list:
            sql = "INSERT INTO t_event_model_file(file_id, file_url, model_id) " \
                  "VALUES(%(file_id)s, %(file_url)s, %(model_id)s)"
            cursor.execute(sql, {'file_id': gi.gener_id_by_uuid(), 'file_url': element, 'model_id': model_id})

        conn.commit()
    except Exception as e:
        conn.rollback()
        logging.error(e)

    conn.close()


def update_task_status(task_id, status="3"):

    sql = "UPDATE t_event_task SET status = %(status)s WHERE model_id = %(model_id)s"
    conn = get_pgsql_connect(schema1, user1, password1, host1, port1)
    cursor = conn.cursor()
    cursor.execute(sql, {'status': status, 'model_id': task_id})
    conn.commit()
    conn.close()


def predict_task_done(task_id, task_result, status="3"):

    conn = get_pgsql_connect(schema1, user1, password1, host1, port1)
    try:
        cursor = conn.cursor()
        sql = "UPDATE t_event_task SET status = %(status)s WHERE model_id = %(model_id)s"
        cursor.execute(sql, {'status': status, 'model_id': task_id})

        sql = "INSERT INTO t_event_task_result(task_result_id, task_result_content, model_id) " \
              "VALUES(%(task_result_id)s, %(task_result_content)s, %(model_id)s)"
        cursor.execute(sql, {'task_result_id': gi.gener_id_by_uuid(),
                             'task_result_content': task_result, 'model_id': task_id})

        conn.commit()
    except Exception as e:
        conn.rollback()
        logging.error(e)

    conn.close()

