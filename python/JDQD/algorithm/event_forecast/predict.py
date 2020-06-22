# -*- coding: utf-8 -*-
import sys
import os
import numpy as np
from datetime import timedelta

module_dir = os.path.dirname(__file__)
import_dir = os.path.join(module_dir, '../..')
log_path = os.path.join(import_dir, 'log/logs')
sys.path.append(import_dir)

from algorithm.event_forecast import model_evalution as mu
import algorithm.event_forecast.preprocess as pp
import utils.pgsql_util as pgsql
import utils.logger as logger

LOG = logger.Logger("debug", log_path=log_path)


class SessionResult:
    def __init__(self, model_dir, load_model=True):
        """
        :param model_dir: 存放模型的目录名
        :param load_model: 是否加载已有模型
        """
        if load_model:
            self.encoder, self.decoder = mu.load_models(model_dir)

    def predict_sample(self, encoder, decoder, input_sample, n_classes, output_len):
        state = encoder.predict(np.array([input_sample]))
        target_seq = np.array([0.0 for _ in range(n_classes)]).reshape([1, 1, n_classes])
        output = []
        for t in range(output_len):
            yhat, h, c = decoder.predict([target_seq] + state)
            output.append(yhat[0, 0, :])
            state = [h, c]
            target_seq = yhat
        return output

    def pred_with_reloaded_model(self, inputs, n_classes, output_len):
        """
        使用加载的模型预测测试集数据
        :param inputs:
        :return:
        """
        preds = [self.predict_sample(self.encoder, self.decoder, inputs_sample, n_classes, output_len) for inputs_sample
                 in inputs]
        return preds


def predict(model_dir, inputs, output_len, n_classes):
    """
    执行页面的预测请求
    :param n_classes:
    :param inputs:
    :param output_len:
    :param model_dir: 模型存放的目录
    :return: 预测结果拼接成的字符串
    """

    session_result = SessionResult(model_dir)
    preds = session_result.pred_with_reloaded_model(inputs, n_classes, output_len)
    return preds


models_dir = '../algorithm/event_forecast/resources/models'


def predict_by_sub_models(data, dates, detail_ids, sub_models, pred_start_date, num_classes):
    preds_one_day = []
    preds_all_days = []
    dates_pred = []
    dates_data_pred = []
    dates_pred_all = []
    predicted_detail_id_dates = pgsql.query_pred_rsts_by_detail_ids_and_pred_start_date(detail_ids, pred_start_date)
    for detail_id, sub_model in zip(detail_ids, sub_models):
        LOG.info(f'正在使用模型{sub_model}进行预测')
        params = sub_model.split('-')[-3:]
        input_len, output_len, n_pca = [int(p) for p in params]
        model_dir = f'{models_dir}/{sub_model}'
        values_pca = pp.apply_pca(n_pca, models_dir, data, True)
        inputs_test, output_dates = pp.gen_inputs_by_pred_start_date(values_pca, input_len, dates, pred_start_date)
        max_output_date = output_dates[-1] + timedelta(1)
        output_dates.append(max_output_date)

        dates_data = [od - timedelta(1) for od in output_dates]

        predicted_dates = predicted_detail_id_dates.get(detail_id)
        if predicted_dates is None:
            latest_date_predicted = False
        else:
            predicted_dates = sorted([pp.parse_date_str(d) for d in predicted_dates])[:-output_len + 1]
            max_predicted_date = predicted_dates[-1]
            zipped_unpredicted = [[d, i, dd] for d, i, dd in zip(output_dates, inputs_test, dates_data) if d not in predicted_dates]
            if not zipped_unpredicted:
                LOG.info(f'{sub_model}所有日期已预测, 跳过')
                continue
            output_dates, inputs_test, dates_data, = zip(*zipped_unpredicted)
            output_dates = list(output_dates)
            inputs_test = list(inputs_test)
            dates_data = list(dates_data)
            if max_predicted_date == max_output_date:
                latest_date_predicted = True
            else:
                latest_date_predicted = False

        dates_all_m = []
        for dd in dates_data:
            dates_all = [dd + timedelta(t) for t in range(1, output_len + 1)]
            dates_all_m.append(dates_all)

        pred = predict(model_dir, inputs_test, output_len, num_classes)
        pred_ = [p[0] for p in pred]
        if not latest_date_predicted:
            pred_.extend(pred[-1][1:])
            output_dates.extend([max_output_date + timedelta(d) for d in range(1, output_len)])
            dates_data.extend(dates_data[-output_len + 1:])

        preds_one_day.append(pred_)
        preds_all_days.append(pred)
        dates_pred.append(output_dates)
        dates_data_pred.append(dates_data)
        dates_pred_all.append(dates_all_m)
    return preds_one_day, preds_all_days, dates_pred, dates_pred_all, dates_data_pred


if __name__ == '__main__':
    pass
