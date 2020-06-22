import numpy as np
import sys
from datetime import date, datetime
import pandas as pd
import os

module_dir = os.path.dirname(__file__)
import_dir = os.path.join(module_dir, '../..')
sys.path.append(import_dir)
from utils import db_conn as db


def obtain_data_from_db(data_tables, con):
    """
    从数据库中查询数据
    :param data_tables: 使用的数据表列表
    :return: 数据库查询结果
    """
    rsts = []
    for t in data_tables:
        rst = db.query_table(t, con)
        rst = [list(r) for r in rst]
        rsts.append(rst)
    return rsts


def obtain_data_from_file():
    data_file1 = f'{module_dir}/resources/data_1.txt'
    data_file2 = f'{module_dir}/resources/data_2.txt'
    data1 = pd.read_csv(data_file1, header=None).values
    data2 = pd.read_csv(data_file2, header=None).values
    data1[data1 == ' NULL'] = 0
    data2[data2 == ' NULL'] = 0
    return [data1, data2]


def obtain_events_from_db(event_col, date_col, con):
    """
    从事件表中获取事件数据
    :return: 事件日期列表, 事件名称列表
    """
    events_table = 'dashijian'
    events = db.query_table(events_table, con)
    events = [[e[event_col].replace(',', ''), e[date_col]] for e in events if e[7] is not None]
    events = [e for e in events if e[0] is not None and e[1] is not None]
    date_type = type(events[0][1])
    if date_type == str:
        events = [[e[0], datetime(*[int(s) for s in e[1].split('-')])] for e in events if len(e[1]) > 7]

    events.sort(key=lambda x: x[1], reverse=True)
    events = np.array(events, dtype=object)
    events = [[e[0], e[1].date()] for e in events]
    return events


def obtain_events_from_file(event_col, date_col):
    """
    Obtain events sequence of types and corresponding dates.
    Note: event types are automatically converted to int type.
    :param event_col:
    :param date_col:
    :return:
    """
    events_file = f'{module_dir}/resources/data_x.txt'
    events = pd.read_csv(events_file, header=None).values[:, [event_col, date_col]]
    events = [[e[0], e[1].replace(' ', '').split('-')] for e in events]
    events = [e for e in events if len(e[1]) == 3 and len(e[1][-1]) > 0]
    events = np.array([[e[0], date(*[int(d) for d in e[1]])] for e in events])
    return events


def table_rst_arrange(rst, date_col, from_txt=False):
    """
    将从数据库中查询得到的数据整理为模型的输入数据
    :param from_txt:
    :param rst: 数据库中查询得到的数据列表
    :param date_col: 日期所在列
    :return: 输入数据的日期列表, 数据表对应的输入数据
    """
    if not from_txt:
        rst = [r[:-1] for r in rst]
        rst = np.array(rst)
    rst = rst[rst[:, date_col].argsort()]
    dates = rst[:, date_col]
    if from_txt:
        dates = [date(*(int(t) for t in d.split(' ')[0].split('-'))) for d in dates]
    else:
        dates = [d.date() for d in dates]
    data = rst[:, 1:]
    data[data == None] = 0
    data = data.astype(int)
    return dates, data


def combine_data(data_tables, con, from_file=False):
    """
    将多个数据表得到的输入数据合并
    :param con:
    :param from_file: 是否从本地文件读取
    :param data_tables: 使用的数据表列表
    :return: 输入数据的日期列表, 合并后的输入数据
    """
    if from_file:
        rsts = obtain_data_from_file()
    else:
        rsts = obtain_data_from_db(data_tables, con)
    data = []
    for r in rsts:
        dates, data_table = table_rst_arrange(r, 0, from_file)
        data.append(data_table)
    data = np.concatenate(data, axis=1).astype(np.float)
    zero_var_cols = [i for i in range(data.shape[1]) if len(set(data[:, i])) == 1]
    data = np.array([data[:, i] for i in range(data.shape[1]) if i not in zero_var_cols])
    data = data.T
    return dates, data


def remove_dupli_dates_events(events, event_priority):
    """
    去除一天内多次发生的重复事件
    :param event_priority:
    :param events: 事件数据
    :return: 去除重复后的事件日期列表, 事件数据
    """
    date_event_dict = {}
    for e, d in events:
        date_event_dict.setdefault(d, []).append(e)

    for d, es in date_event_dict.items():
        if len(es) > 0:
            if event_priority in es:
                date_event_dict[d] = [event_priority]
            else:
                date_event_dict[d] = es[:1]

    events = list(date_event_dict.items())
    dates_events = [e[0] for e in events]
    events = [e[1][0] for e in events]
    return dates_events, events


def padding_events(dates_x, dates_events, events):
    """
    将事件数据中没发生事件的日期跟据对应的输入数据日期使用0进行填充
    :param dates_x: 输入数据日期列表
    :param dates_events: 事件数据的日期列表
    :param events: 出去重复后的事件数据
    :return: 填充后的事件数据
    """
    events_p = []
    events_dtype = type(events[0])
    for i, dx in enumerate(dates_x):
        if dx in dates_events:
            events_p.append(events[dates_events.index(dx)])
        else:
            events_p.append(events_dtype(0))
    return np.array(events_p)


def get_events(dates, event_priority, from_file=False, con='postgresql', event_col=1, date_col=5):
    if from_file:
        events = obtain_events_from_file(event_col, date_col)
    else:
        events = obtain_events_from_db(event_col, date_col, con)
    dates_events, events = remove_dupli_dates_events(events, event_priority)
    events_p = padding_events(dates, dates_events, events)
    return events_p
