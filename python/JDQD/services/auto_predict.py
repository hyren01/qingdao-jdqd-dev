# -*- coding: utf-8 -*-
import sys
from keras import backend as K
import time
import numpy as np
import os

module_dir = os.path.dirname(__file__)
import_dir = os.path.join(module_dir, '..')
log_path = os.path.join(import_dir, 'log/logs')
sys.path.append(import_dir)
from config import server_cfg
import utils.logger as logger
import utils.pgsql_util as pgsql
import algorithm.event_forecast.obtain_data as od
import algorithm.event_forecast.preprocess as pp
from algorithm.event_forecast import predict

LOG = logger.Logger("debug", log_path=log_path)

con = 'postgresql'
super_event_col = server_cfg.super_event_col
sub_event_col = server_cfg.sub_event_col
date_col = server_cfg.date_col
event_priority = server_cfg.event_priority


def model_predict(model_id, task_id, tables, pred_start_date):
    tables = np.array(str(tables).split(","))

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
        LOG.error("表 {} 预测发生异常：{}".format(','.join(tables), e))
        return {"success": False, "exception": e}
    finally:
        K.clear_session()


if __name__ == '__main__':
    argvs = sys.argv
    task_id, model_id, tables, pred_start_date = argvs[1:]
    model_predict(model_id, task_id, tables, pred_start_date)



