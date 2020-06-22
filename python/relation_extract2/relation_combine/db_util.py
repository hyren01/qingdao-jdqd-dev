import psycopg2


def get_conn(database, user, password, host, port):
    conn = psycopg2.connect(database=database, user=user, password=password, host=host, port=port)
    print('connection ok')
    return conn


def query(conn, sql):
    cursor = conn.cursor()
    cursor.execute(sql)
    results = cursor.fetchall()
    conn.close()
    return results


def modify(conn, sql, error=''):
    if sql:
        cursor = conn.cursor()
        try:
            sqls = [sql] if type(sql) == str else sql
            for s in sqls:
                cursor.execute(s)
            conn.commit()
        except Exception as e:
            conn.rollback()
            error = f'{error}: ' if error else error
            print(f'{error}{e}')
        finally:
            conn.close()
