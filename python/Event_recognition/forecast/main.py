# -*- coding: utf-8 -*-

import logging
import os
import numpy as np
import pandas as pd
import helper.pgsql_helper as input_resource
import flask
import utils.gener_id as gi
from flask import request
from config import Config

webApp = flask.Flask(__name__)
webApp.config.update(RESTFUL_JSON=dict(ensure_ascii=False))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s -%(message)s")

conf = Config()
model_root_path = conf.model_root_path
data_root_path = conf.data_root_path
csv_name = "model_data.csv"


def mkdir_p(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, 0o744, True)


@webApp.route("/buildModel", methods=['GET', 'POST'])
def build_model():
    model_id = request.form.get("model_id")
    model_id = str(model_id)
    tables = request.form.get("tables")
    tables = np.array(str(tables).split(","))
    event_type = request.form.get("event_type")
    days = request.form.get("days")
    days = int(days)
    min_dim = request.form.get("min_dim")   # 最小降维
    min_dim = 5 if min_dim is None else int(min_dim)
    max_dim = request.form.get("max_dim")   # 最大降维
    max_dim = 10 if max_dim is None else int(max_dim)
    min_lag = request.form.get("min_lag")   # 最小滞后期
    min_lag = 10 if min_lag is None else int(min_lag)
    max_lag = request.form.get("max_lag")   # 最大滞后期
    max_lag = 61 if max_lag is None else int(max_lag)
    unit = request.form.get("unit")         # 神经元个数
    unit = 128 if unit is None else int(unit)
    batch = request.form.get("batch")       # 批量数据个数
    batch = 64 if batch is None else int(batch)
    epoch = request.form.get("epoch")       # 训练次数
    epoch = 150 if epoch is None else int(epoch)
    try:
        logging.info("开始构建表 {} 数据".format(','.join(tables)))
        model_data_path = build_data(tables, event_type)
        logging.info("开始根据表 {} 数据构建模型".format(','.join(tables)))
        model_path_list = []
        for element in model_data_path:
            model_path = train_model.execute(data_file=element, muti_day=days, min_dim=min_dim, max_dim=max_dim,
                                             min_lag=min_lag, max_lag=max_lag, unit=unit, batch=batch, epoch=epoch)
            model_path_list.append(model_path)

        for element in model_data_path:
            if os.path.exists(element):
                os.remove(element)

        input_resource.model_train_done(model_id, model_path_list, "2")    # 2表示运行完成
        logging.info("当前表 {} 的模型构建完成".format(','.join(tables)))
        return {"success": True, "model_path": model_path_list}
    except Exception as e:
        input_resource.update_model_status(model_id, "3")   # 3表示运行失败
        logging.error("表 {} 构建发生异常：{}".format(','.join(tables), e))
        return {"success": False, "exception": e}


@webApp.route("/modelPredict", methods=['GET', 'POST'])
def model_predict():
    task_id = request.form.get("model_id")    # 预测任务编号
    task_id = str(task_id)
    models = request.form.get("models")
    models = np.array(str(models).split(","))
    tables = request.form.get("tables")
    tables = np.array(str(tables).split(","))
    days = request.form.get("days")
    days = int(days)
    min_dim = request.form.get("min_dim")  # 最小降维
    min_dim = 5 if min_dim is None else int(min_dim)
    max_dim = request.form.get("max_dim")  # 最大降维
    max_dim = 10 if max_dim is None else int(max_dim)
    min_lag = request.form.get("min_lag")  # 最小滞后期
    min_lag = 10 if min_lag is None else int(min_lag)
    max_lag = request.form.get("max_lag")  # 最大滞后期
    max_lag = 61 if max_lag is None else int(max_lag)
    try:
        logging.info("开始根据表 {} 数据使用模型进行预测".format(','.join(tables)))
        result, column_number = build_basic_data(tables)
        data_path = data_root_path + "/" + gi.gener_id_by_uuid() + ".csv"
        result.to_csv(data_path, index=False, encoding='utf-8', header=None)

        all_preds = []
        for element in models:
            preds = predict.execute(element, data_path, days, min_dim, max_dim, min_lag, max_lag)
            all_preds = all_preds.__add__(preds)
        result_preds = predict.transform_result(all_preds)
        os.remove(data_path)

        input_resource.predict_task_done(task_id, str(result_preds), "2")  # 2表示运行完成
        logging.info("当前表 {} 的模型预测完成".format(','.join(tables)))
        return {"success": True, "predict": result_preds}
    except Exception as e:
        input_resource.update_task_status(task_id, "3")  # 3表示运行失败
        logging.error("表 {} 预测发生异常：{}".format(','.join(tables), e))
        return {"success": False, "exception": e}


def drop_col(df, col_names):
    df_data_nums = len(df)
    for element in col_names:
        col_data_nums = df[element].count()
        if df_data_nums != col_data_nums:
            df.drop(element, axis=1, inplace=True)


def build_basic_data(tables):
    result = None
    column_start_key = 1
    for index, element in enumerate(tables):
        rows = input_resource.get_predict_rows_by_table(element, True)
        data = np.array(rows)
        """
            假设有4张表，字段个数分别为[5, 6, 7, 8]，以下代码会产生的结果为：
            当第一张表字段个数为5时，dataFrame所产生的字段为：['0', 1, 2, 3, 4]
            当第二张表字段个数为6时，dataFrame所产生的字段为：['0', 5, 6, 7, 8, 9]
            当第三张表字段个数为7时，dataFrame所产生的字段为：['0', 10, 11, 12, 13, 14, 15]
            当第四张表字段个数为8时，dataFrame所产生的字段为：['0', 16, 17, 18, 19, 20, 21, 22]
            以上结果是基于dataFrame要进行merge的原因，merge要指定key，以及列名不能重复，该处的key都为'0'列
        """
        columns = ['0'].__add__(list(range(column_start_key, column_start_key + len(data[0]) - 1)))
        column_start_key = column_start_key + len(data[0]) - 1
        df = pd.DataFrame(data=data, columns=columns)
        drop_col(df, columns)  # 清理空值列
        if result is None:
            result = df
            result["0"] = result["0"].astype("str")
        else:
            df["0"] = df["0"].astype("str")
            result = pd.merge(result, df, on='0')

    return result, column_start_key


def build_data(predict_tables, event_type):

    model_data_path = []
    result, column_number = build_basic_data(predict_tables)
    normal_rows, duplicate_rows = input_resource.get_event_rows(event_type)
    data = np.array(normal_rows)
    df = pd.DataFrame(data=data, columns=['0', column_number])
    df["0"] = df["0"].astype("str")
    result_no_duplicate = pd.merge(result, df, on='0')

    uuid = gi.gener_id_by_uuid()
    csv_dir = model_root_path + "/" + uuid + "/main"
    mkdir_p(csv_dir)
    csv_path = csv_dir + "/" + csv_name
    result_no_duplicate.to_csv(csv_path, index=False, encoding='utf-8', header=None)
    model_data_path.append(csv_path)
    logging.info("表 {} 合并为无重复数据的CSV文件成功，路径在 {}".format(','.join(predict_tables), csv_path))

    for element in duplicate_rows:
        event_date = element.get("rqsj")
        if result[result['0'].isin([event_date])].__len__() == 0:
            continue
        event_id = element.get("event_id")
        rows_by_duplicate = input_resource.get_event_rows_by_duplicate_data(event_date, event_id, event_type)
        data = np.array(rows_by_duplicate)
        columns = ['0', column_number]
        df = pd.DataFrame(data=data, columns=columns)
        drop_col(df, columns)  # 清理空值列
        df["0"] = df["0"].astype("str")
        result_by_duplicate = pd.merge(result, df, on='0')

        csv_dir = model_root_path + "/" + uuid + "/" + event_id
        mkdir_p(csv_dir)
        csv_path = csv_dir + "/" + csv_name
        result_by_duplicate.to_csv(csv_path, index=False, encoding='utf-8', header=None)
        model_data_path.append(csv_path)
        logging.info("表 {} 由重复数据合并为CSV文件成功，事件日期为 {}，事件编号为 {}，路径在 {}"
                     .format(','.join(predict_tables), event_date, event_id, csv_path))

    return model_data_path


if __name__ == '__main__':
    from services import train_model
    from services import predict
    webApp.config['JSON_AS_ASCII'] = False
    webApp.run(host='0.0.0.0', port=38080, debug=True)
