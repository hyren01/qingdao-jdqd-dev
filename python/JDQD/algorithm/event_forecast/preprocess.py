# coding=utf-8
import numpy as np
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from datetime import date, timedelta
import os

module_dir = os.path.dirname(__file__)
import_dir = os.path.join(module_dir, '../..')


def one_hot_dict(events_set):
    """
    根据事件列表生成每个事件类型与对应one-hot形式向量的字典
    :param events_set: 事件列表
    :return: 每个事件类型与对应one-hot形式向量的字典
    """
    m = len(events_set)
    eye = np.eye(m)
    oh_dict = {}
    for i, r in zip(events_set, eye):
        oh_dict[i] = r
    return oh_dict


def get_events_set(events_p):
    return sorted(set(events_p))


def events_one_hot(events_p, events_set):
    """
    将事件列表转化成one-hot形式表示的矩阵
    :param events_set: 事件
    :param events_p: 补全操作后的事件列表
    :return: one-hot形式表示的事件类型矩阵
    """
    oh_dict = one_hot_dict(events_set)
    events_p_oh = np.array([oh_dict[a] for a in events_p])
    return events_p_oh


def events_binary(events_p_oh, event_col):
    """
    将one-hot形式的事件矩阵转换成使用[0, 1]表示的某个特定事件是否发生的列表
    :param events_p_oh: one-hot形式的事件矩阵
    :param event_col: 需转换的事件在one-hot矩阵中对应的列数
    :return: 使用[0, 1]表示的某个特定事件是否发生的列表
    """
    return np.equal(events_p_oh[:, event_col], 1).astype(int)


def get_recur_events_rows(events_p_oh):
    """
    获取多次出现的事件与其对应发生的日期在输入数据中的行数列表的字典
    :param events_p_oh: one-hot形式的事件矩阵
    :return: 多次出现的事件与其对应发生的日期在输入数据中的行数列表的字典
    """
    recur_events_cols = np.where(np.sum(events_p_oh[..., 1:], axis=0) > 1)[0] + 1
    recur_events_rows = {c: np.where(events_p_oh[:, c] == 1)[0] for c in recur_events_cols}
    return recur_events_rows


def apply_pca(pca_n, pca_dir, data, reload=False):
    """
    对输入数据进行pca降维
    :param data: 数据表数据
    :param pca_n: 降维后维度
    :param pca_dir: pca模型所在目录
    :param reload: 是否加载已存在的pca模型文件
    :return: 降维操作后的数据, 数据日期列表
    """
    if reload:
        data_pca = joblib.load(f"{pca_dir}/{pca_n}fit_pca").transform(data)
    else:
        pca = PCA(n_components=pca_n)
        data_pca = pca.fit_transform(data)
        joblib.dump(pca, f'{pca_dir}/{pca_n}fit_pca')
    return data_pca


def input_end_row_to_start_row(end_row, input_len):
    return end_row - input_len + 1


def format_date(date_):
    """
    convert python built-in datetime.date datatype to str of format yyyy-mm-dd
    :param date_:
    :return:
    """
    return date_.strftime('%Y-%m-%d')


def parse_date_str(date_str):
    """
    convert date string of 'yyyy-mm-dd' format to
    python's built-in datetime.date datatype
    :param date_str:
    :return:
    """
    date_ = date_str.split('-')
    date_ = [int(d) for d in date_]
    date_ = date(*date_)
    return date_


def get_prev_date(cur_date, prev_days):
    cur_date = parse_date_str(cur_date)
    prev_date = cur_date - timedelta(prev_days)
    return format_date(prev_date)


def date_to_row(date_str, dates):
    date_ = parse_date_str(date_str)
    if date_ not in dates:
        return None
    return dates.index(date_)


def row_to_date(row, dates):
    return dates[row]


def gen_input_by_start_row(values_pca, start_row, input_len):
    input_sample = values_pca[start_row: start_row + input_len]
    return input_sample


def gen_output_by_start_row(events_p_oh, start_row, input_len, output_len):
    output_sample = events_p_oh[start_row + input_len: start_row + input_len + output_len]
    return output_sample


def flatten_outputs(outputs):
    outputs_flatten = [o[0] for o in outputs]
    outputs_flatten.extend(outputs[-1][1:])
    return np.array(outputs_flatten)


def get_event_num(outputs, events_set):
    outputs_flatten = flatten_outputs(outputs)
    events_num = {}
    for i, e in enumerate(events_set):
        n_event = int(np.sum(outputs_flatten[:, i]))
        events_num[e] = n_event
    return events_num


def gen_samples_by_date(values_pca, events_p_oh, input_len, output_len, dates, pred_start_date, pred_end_date):
    """
    获取起始-终止预测日期内对应的输入数据及输出数据
    :param values_pca:
    :param events_p_oh:
    :param input_len:
    :param output_len:
    :param dates:
    :param pred_start_date:
    :param pred_end_date:
    :return:
    """
    min_output_start_row = date_to_row(pred_start_date, dates)
    max_output_end_row = date_to_row(pred_end_date, dates)
    max_output_end_row = min(max_output_end_row, len(dates) - 1)
    min_input_start_row = min_output_start_row - input_len
    min_input_start_row = max(min_input_start_row, 0)
    max_input_end_row = max_output_end_row - output_len
    max_input_start_row = input_end_row_to_start_row(max_input_end_row, input_len)

    inputs = [gen_input_by_start_row(values_pca, start_row, input_len)
              for start_row in range(min_input_start_row, max_input_start_row + 1)]

    outputs = [gen_output_by_start_row(events_p_oh, start_row, input_len, output_len)
               for start_row in range(min_input_start_row, max_input_start_row + 1)]

    return np.array(inputs), np.array(outputs)


def gen_latest_inputs(values_pca, input_len):
    """
    生成预测输入数据, 根据输入序列长度选取对应长度的最新的数据
    :param values_pca:
    :param input_len:
    :return:
    """
    max_start_row = len(values_pca) - input_len
    input_ = [gen_input_by_start_row(values_pca, max_start_row, input_len)]
    return np.array(input_)


def gen_inputs_by_pred_start_date(values_pca, input_len, dates, pred_start_date):
    """
    生成预测输入数据, 根据输入序列长度选取对应长度的指定日期之后的数据
    :param dates:
    :param pred_start_date:
    :param values_pca:
    :param input_len:
    :return:
    """
    pred_start_row = date_to_row(pred_start_date, dates)
    min_input_start_row = max(pred_start_row - input_len, 0)
    max_input_start_row = len(values_pca) - input_len
    input_ = [gen_input_by_start_row(values_pca, row, input_len)
              for row in range(min_input_start_row, max_input_start_row + 1)]
    dates_ = dates[pred_start_row:]
    return np.array(input_), dates_

#
# def gen_inputs_by_pred_first_date(values_pca, input_len, dates, pred_first_date):
#     pred_first_date_row = date_to_row(pred_first_date, dates)
#     input_start_row = pred_first_date_row - input_len
#     if input_start_row < 0:
#         return None
#     input_ = gen_input_by_start_row(values_pca, input_start_row, input_len)
#     return input_


if __name__ == '__main__':
    pass
    import obtain_data as od
    #
    # dates, data = od.combine_data(None, None, True)
    # events_p = od.get_events(dates, 11209, from_file=True, event_col=1, date_col=5)
    # events_set = get_events_set(events_p)
    # events_p_oh = events_one_hot(events_p, events_set)

