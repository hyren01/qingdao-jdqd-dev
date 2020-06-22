import psycopg2
import json
import os

module_path = os.path.dirname(__file__)


def load_cfg():
    db_cfg_f = '../config/dbconn_cfg'
    with open(db_cfg_f, 'r') as f:
        l = f.read()
    db_config = json.loads(l)
    return db_config


def get_conn():
    db_config = load_cfg()
    conn = psycopg2.connect(database=db_config['DATABASE'], user=db_config['DBUSER'], password=db_config['PASSWORD'],
                            host=db_config['HOST'], port=db_config['PORT'])
    conn.set_client_encoding('utf-8')
    return conn


def get_conn_g():
    from GBaseConnector import connect
    db_config = load_cfg()
    conn = connect()
    conn.connect(**db_config)
    return conn


def get_conn_m():
    import pymysql
    db_config = load_cfg()
    conn = pymysql.connect(host=db_config['HOST'], port=db_config['PORT'], user=db_config['DBUSER'],
                           password=db_config['PASSWORD'], db=db_config['DATABASE'])
    return conn


def query(sql, con):
    if con == 'postgresql':
        conn = get_conn()
    elif con == 'gbase':
        conn = get_conn_g()
    elif con == 'mysql':
        conn = get_conn_m()
    else:
        conn = None
    cursor = conn.cursor()
    cursor.execute(sql)
    rst = cursor.fetchall()
    cursor.close()
    conn.close()
    return rst


def query_table(table_name, con):
    sql = 'select * from {}'.format(table_name)
    return query(sql, con)
