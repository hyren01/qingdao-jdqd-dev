import psycopg2
import json
from psycopg2.errors import UniqueViolation
from urllib.parse import urlencode
from urllib.request import urlopen
import time

IS_DEBUG = True


def debug(*msg):
    if IS_DEBUG:
        size = len(msg)
        if msg is None or size < 1:
            print("")
            return

        print("[DEBUG] ", end="")
        for s in msg:
            print(s, " ", end="")
        print("")

def http_post(data, uri):

    data = urlencode(data)
    data = data.encode()
    res = urlopen(url=uri, data=data)
    content = res.read()

    return content


debug("\n")
conn = psycopg2.connect(database="ebmdb", user="jdqd", password="jdqd", host="139.9.126.19", port="31001")
#debug(f"Open  connection : {conn}")
cur = conn.cursor()
#debug(f"Get   cursor     : {cur}")

try:
    url = "http://172.168.0.115:38082/coref_with_content"
    cur.execute("SELECT * FROM t_article_msg_en where substring(article_id,1,1) ='a' and article_id not in ( select distinct article_id from  t_article_msg_zh) and content <>''")
    rs = cur.fetchall()
    for i, row in enumerate(rs):
        # debug(row[0], row[1][:100])

        # 调用指代消解接口
        #data = bytes(urlencode({'content': f"{row[1]}"}), encoding='utf-8')
        data = {'content': f"{row[1]}"}
        result = http_post(data,url)
        result = json.loads(result)
        # result = post(, f"content={row[1]}")
        # 数据插入表
        debug(f"deal article_id : {row[0]}")
        debug(f"deal data : {i}")
        try:
            cur.execute("INSERT INTO t_article_msg_zh(article_id, content) VALUES(%s,%s)",
                        (row[0], result["coref"]))
        except Exception as ex:
            if isinstance(ex, UniqueViolation):
                debug(f"Key already exists! article_id={row[0]}")
            else:
                print(f"[ERROR]:[id={row[0]}] ==== ex.type={type(ex)}, ex.msg={ex}")
            conn.rollback()
        # if i % 1000 == 0:
        #     conn.commit()
        #     cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        #     debug(f"cur_time : {cur_time}")
        conn.commit()
        cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        debug(f"endcur_time : {cur_time}")

except Exception as ex:
    print(f"[ERROR] ==== ex.type={type(ex)}, ex.msg={ex}")
finally:
    if cur is not None:
        cur.close()
        debug(f"Close cursor     : {cur}")
    if conn is not None:
        conn.close()
        debug(f"Close connection : {conn}")
