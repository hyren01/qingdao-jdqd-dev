# -*- coding: utf-8 -*-
import sys
import flask
from flask import request
from keras import backend as K
import numpy as np
import time
import os
import traceback

module_dir = os.path.dirname(__file__)
import_dir = os.path.join(module_dir, '..')
log_path = os.path.join(import_dir, 'log/logs')
sys.path.append(import_dir)

from config import server_cfg
import utils.logger as logger
import utils.pgsql_util as pgsql
import algorithm.event_forecast.obtain_data as od
import algorithm.event_forecast.preprocess as pp
import algorithm.event_forecast.model_evalution as meval

webApp = flask.Flask(__name__)
webApp.config.update(RESTFUL_JSON=dict(ensure_ascii=False))

LOG = logger.Logger("debug", log_path=log_path)

con = 'postgresql'
w1 = server_cfg.w1
w2 = server_cfg.w2
w3 = server_cfg.w3
w4 = server_cfg.w4
super_event_col = server_cfg.super_event_col
sub_event_col = server_cfg.sub_event_col
date_col = server_cfg.date_col
event_priority = server_cfg.event_priority


@webApp.route("/buildModel", methods=['GET', 'POST'])
def build_model():
    model_id = request.form.get("model_id")
    model_name = request.form.get("model_name")
    tables = request.form.get("tables")
    tables = np.array(str(tables).split(","))
    event_type = request.form.get("event_type")
    output_len = request.form.get("days")
    output_len = int(output_len)

    min_dim = request.form.get("min_dim")  # 最小降维
    min_dim = 5 if min_dim is None else int(min_dim)
    max_dim = request.form.get("max_dim")  # 最大降维
    max_dim = 10 if max_dim is None else int(max_dim)

    min_input_len = request.form.get("min_lag")  # 最小滞后期
    min_input_len = int(min_input_len) if min_input_len else 10
    max_input_len = request.form.get("max_lag")  # 最大滞后期
    max_input_len = int(max_input_len) if max_input_len else 61

    num_units = request.form.get("unit")  # 神经元个数
    num_units = int(num_units) if num_units else 128

    batch = request.form.get("batch")  # 批量数据个数
    batch = int(batch) if batch else 64

    epoch = request.form.get("epoch")  # 训练次数
    epoch = int(epoch) if epoch else 150
    event_col = super_event_col if event_type == '2' else sub_event_col

    step = request.form.get('size')
    step = int(step) if step else 4

    train_start_date = request.form.get('tran_start_date')
    train_end_date = request.form.get('tran_end_date')
    eval_start_date = request.form.get('evaluation_start_date')
    eval_end_date = request.form.get('evaluation_end_date')

    dates, data = od.combine_data(tables, con, from_file=server_cfg.load_data_from_file)
    events_p = od.get_events(dates, event_priority, from_file=server_cfg.load_data_from_file, event_col=event_col,
                             date_col=date_col)
    events_set = pp.get_events_set(events_p)
    events_p_oh = pp.events_one_hot(events_p, events_set)
    n_classes = len(events_set)

    try:
        sub_model_dirs, sub_model_names, outputs_list, params_list = train_model.execute(model_name, data, events_p_oh,
                                                                                         dates, train_start_date,
                                                                                         train_end_date,
                                                                                         output_len, min_dim, max_dim,
                                                                                         min_input_len,
                                                                                         max_input_len, step, num_units,
                                                                                         batch, epoch)

        LOG.info('训练完成, 模型存入数据库')
        pgsql.model_train_done(model_id, sub_model_dirs)  # 2表示运行完成

        detail_ids = pgsql.insert_into_model_detail(sub_model_names, model_id)

        pgsql.insert_into_model_train(detail_ids, outputs_list, events_set)

        LOG.info('开始评估模型')

        meval.evaluate_sub_models(data, dates, detail_ids, sub_model_dirs, params_list, events_p_oh, events_set, n_classes,
                                  eval_start_date, eval_end_date, w1, w2, w3, w4)

        date_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        date_str, time_str = date_time.split(' ')
        pgsql.model_eval_done(model_id, date_str, time_str, '2')
        LOG.info("当前表 {} 的模型构建完成".format(','.join(tables)))
        return {"success": True, "model_path": sub_model_dirs}
    except Exception as e:
        pgsql.update_model_status(model_id, "3")  # 3表示运行失败
        LOG.error("表 {} 构建发生异常：{}".format(','.join(tables), e))
        return {"success": False, "exception": e}
    finally:
        K.clear_session()


@webApp.route("/modelPredict", methods=['GET', 'POST'])
def model_predict():
    model_id = request.form.get("model_id")
    model_id = str(model_id)
    tables = request.form.get("tables")
    tables = np.array(str(tables).split(","))
    task_id = request.form.get("task_id")
    pred_start_date = request.form.get("sample_start_date")

    LOG.info("开始根据表 {} 数据进行预测".format(','.join(tables)))

    try:
        dates, data = od.combine_data(tables, con, from_file=server_cfg.load_data_from_file)
        events_p = od.get_events(dates, event_priority, from_file=server_cfg.load_data_from_file,
                                 event_col=super_event_col,
                                 date_col=date_col)
        events_set = pp.get_events_set(events_p)
        sub_model_results = pgsql.query_sub_models_by_model_id(model_id)
        sub_models = [r[0] for r in sub_model_results]
        detail_ids = [r[1] for r in sub_model_results]
        num_classes = len(events_set)

        preds, preds_all_days, dates_pred, dates_pred_all, dates_data_pred = predict.predict_by_sub_models(data, dates, detail_ids, sub_models, pred_start_date, num_classes)
        pgsql.insert_pred_result(preds, preds_all_days, dates_pred, dates_pred_all, dates_data_pred, detail_ids, events_set, task_id)

        date_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        date_str, time_str = date_time.split(' ')
        pgsql.predict_task_done(task_id, date_str, time_str, "2")  # 2表示运行完成
        LOG.info("当前表 {} 的模型预测完成".format(','.join(tables)))
        return {"success": True}
    except Exception as e:
        pgsql.update_task_status(task_id, "3")  # 3表示运行失败
        LOG.error(f"表 {','.join(tables)} 预测发生异常：{traceback.format_exc()}")
        return {"success": False, "exception": e}
    finally:
        K.clear_session()


if __name__ == '__main__':
    from algorithm.event_forecast import predict, train_model

    webApp.config['JSON_AS_ASCII'] = False
    webApp.run(host=server_cfg.host, port=server_cfg.port, debug=True)
