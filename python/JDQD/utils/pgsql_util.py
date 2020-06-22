# -*- coding: utf-8 -*-
import psycopg2
import utils.gener_id as gi
import utils.logger as logger
import sys
import numpy as np
import time
import os

module_dir = os.path.dirname(__file__)
import_dir = os.path.join(module_dir, '..')
log_path = os.path.join(import_dir, 'log/logs')
sys.path.append(import_dir)

from algorithm.event_forecast import preprocess as pp
from config import Config

conf = Config()
schema1 = conf.input_jdbc_schema1  # 管理数据所在库
user = conf.input_jdbc_user
password = conf.input_jdbc_password
host = conf.input_jdbc_host
port = conf.input_jdbc_port

LOG = logger.Logger("debug", log_path=log_path)


def get_pgsql_connect(jdbc_schema, jdbc_user, jdbc_password, jdbc_host, jdbc_port):
    return psycopg2.connect(database=jdbc_schema, user=jdbc_user, password=jdbc_password, host=jdbc_host,
                            port=jdbc_port)


def query(sql):
    if not sql:
        return None
    conn = get_pgsql_connect(schema1, user, password, host, port)
    cursor = conn.cursor()
    cursor.execute(sql)
    results = cursor.fetchall()
    conn.close()
    return results


def modify(sql, error=''):
    if sql:
        conn = get_pgsql_connect(schema1, user, password, host, port)
        cursor = conn.cursor()
        try:
            sqls = [sql] if type(sql) == str else sql
            for s in sqls:
                cursor.execute(s)
            conn.commit()
        except Exception as e:
            conn.rollback()
            error = f'{error}: ' if error else error
            LOG.error(f'{error}{e}')
        finally:
            conn.close()


def update_model_status(model_id, status="3"):
    sql = f"UPDATE t_event_model SET status = '{status}' WHERE model_id = '{model_id}'"
    modify(sql)


def model_train_done(model_id, file_path_list):
    sqls = []
    for element in file_path_list:
        sql = f"INSERT INTO t_event_model_file(file_id, file_url, model_id) " \
              f"VALUES('{gi.gener_id_by_uuid()}', '{element}', '{model_id}') "
        sqls.append(sql)
    modify(sqls)


def model_eval_done(model_id, date_str, time_str, status="3"):
    sql = f"UPDATE t_event_model SET status = '{status}', tran_finish_date = '{date_str}', " \
          f"tran_finish_time = '{time_str}' WHERE model_id = '{model_id}'"
    modify(sql)


def insert_into_model_detail(sub_model_names, model_id):
    detail_ids = []
    status = 1
    sqls = []
    for sub_model_name in sub_model_names:
        detail_id = gi.gener_id_by_uuid()
        detail_ids.append(detail_id)
        sql = f"insert into t_event_model_detail(detail_id, model_name, status, model_id) values\
              ('{detail_id}', '{sub_model_name}', '{status}', '{model_id}')"
        sqls.append(sql)
    modify(sqls)
    return detail_ids


def insert_into_model_train(detail_ids, outputs_list, events_set):
    status = 1
    sqls = []
    for detail_id, outputs in zip(detail_ids, outputs_list):
        events_num = pp.get_event_num(outputs, events_set)
        for e in events_set:
            tran_id = gi.gener_id_by_uuid()
            event_num = events_num[e]
            sql = f"insert into t_event_model_tran(tran_id, event_name, num, detail_id, status) \
                  values('{tran_id}', '{e}', '{event_num}', '{detail_id}', '{status}')"
            sqls.append(sql)
    modify(sqls)


def insert_model_test(event, event_num, false_rate, recall_rate, false_alarm_rate, tier_precision, tier_recall,
                      bleu, detail_id):
    test_id = gi.gener_id_by_uuid()
    status = 1
    sql = f"insert into t_event_model_test(test_id, event_name, num, false_rate, recall_rate, \
                         false_alarm_rate, tier_precision, tier_recall, bleu, status, detail_id) \
                         values('{test_id}', '{event}', '{event_num}', '{false_rate}', '{recall_rate}', '{false_alarm_rate}', \
                         '{tier_precision}', '{tier_recall}', '{bleu}', '{status}', '{detail_id}')"
    error = '插入子模型评估结果失败'
    modify(sql, error)


def insert_model_tot(scores, events_num):
    status = 1
    scores.sort(key=lambda x: x[0], reverse=True)
    top_scores = scores[:min(10, len(scores))]
    LOG.info('top模型存入数据库')
    sqls = []
    for score, bleu_summary, tier_precision_summary, tier_recall_summary, fr_summary, rc_summary, fa_summary, detail_id in top_scores:
        num_events = np.sum([v for k, v in events_num.items() if str(k) != '0'])
        tot_id = gi.gener_id_by_uuid()
        sql_summary = f"insert into t_event_model_tot(tot_id, num, false_rate, recall_rate, false_alarm_rate, " \
                      f"tier_precision, tier_recall, bleu, score, status, detail_id) " \
                      f"values('{tot_id}', '{num_events}', '{fr_summary}', '{rc_summary}', '{fa_summary}', " \
                      f"'{tier_precision_summary}', '{tier_recall_summary}', '{bleu_summary}', '{score}', " \
                      f"'{status}', '{detail_id}')"
        sqls.append(sql_summary)
    modify(sqls, '存入top模型出错')


def query_sub_models_by_model_id(model_id):
    sql = f"select detail_id, score from t_event_model_tot where detail_id in " \
          f"(select detail_id from t_event_model_detail where model_id = '{model_id}') "
    results1 = query(sql)
    detail_ids = [f"'{r[0]}'" for r in results1]
    detail_ids = f"({','.join(detail_ids)})"
    sql2 = f"select model_name, detail_id from t_event_model_detail where detail_id in {detail_ids}"
    results2 = query(sql2)
    return results2


def query_pred_rsts_by_detail_ids_and_pred_start_date(detail_ids, pred_start_date):
    detail_ids = [f"'{i}'" for i in detail_ids]
    detail_ids = f"({','.join(detail_ids)})"
    sql = f"select detail_id, forecast_date from t_event_task_rs " \
          f"where detail_id in {detail_ids} and forecast_date >= '{pred_start_date}'"
    results = query(sql)
    detail_id_dates = {}
    for r in results:
        detail_id_dates.setdefault(r[0], set()).add(r[1])
    return detail_id_dates



def insert_new_pred_task(task_id, model_id, model_name, tables_name, epoch, time_str):
    sql = f"insert into t_event_task(task_id, model_id, model_name, tables_name, epoch, create_time) " \
          f"values('{task_id}', '{model_id}', '{model_name}', '{tables_name}', '{epoch}', '{time_str}')"
    modify(sql)


def insert_pred_result(probs, probs_all_days, dates, dates_pred_all, dates_data, detail_ids, events_set, task_id):
    sqls = []
    sqls_his = []
    for p, pa, ds, da, did, dd in zip(probs, probs_all_days, dates, dates_pred_all, detail_ids, dates_data):
        for i, e in enumerate(events_set):
            for j, d in enumerate(ds):
                rs_id = gi.gener_id_by_uuid()
                date_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                date_str, time_str = date_time.split(' ')
                sql_task_rs = f"insert into t_event_task_rs(rs_id, event_name, probability, forecast_date, status, " \
                              f"detail_id, task_id, create_date, create_time, predict_end_date) values('{rs_id}', '{e}', " \
                              f"'{p[j][i]:.4f}', '{d}', '1', '{did}', '{task_id}', '{date_str}', '{time_str}', '{dd[j]}')"
                sqls.append(sql_task_rs)
            for j, d in enumerate(da):
                pd = pa[j]

                for k, d_ in enumerate(d):

                    pd_ = pd[k]

                    rs_id = gi.gener_id_by_uuid()
                    date_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                    date_str, time_str = date_time.split(' ')
                    sql_task_rs_his = f"insert into t_event_task_rs_his(rs_id, event_name, probability, forecast_date, " \
                                  f"detail_id, task_id, create_date, create_time, predict_end_date) values('{rs_id}', '{e}', " \
                                  f"'{pd_[i]:.4f}', '{d_}', '{did}', '{task_id}', '{date_str}', '{time_str}', '{dd[j]}')"
                    sqls_his.append(sql_task_rs_his)
    error = f't_event_task_rs表预测结果插入出错'
    modify(sqls, error)

    error_his = f't_event_task_rs_his表预测结果插入出错'
    modify(sqls_his, error_his)



def update_task_status(task_id, status="3"):
    sql = f"UPDATE t_event_task SET status = '{status}' WHERE task_id = '{task_id}'"
    modify(sql)


def predict_task_done(task_id, date_str, time_str, status="3"):
    sql = f"UPDATE t_event_task SET status = '{status}', task_finish_date = '{date_str}', " \
          f"task_finish_time = '{time_str}' WHERE task_id = '{task_id}'"
    modify(sql)


if __name__ == '__main__':
    model_id = '22222'
