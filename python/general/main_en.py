import psycopg2
import time
from utils.translate_util import translate_any_2_anyone
from psycopg2.errors import UniqueViolation

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


conn = psycopg2.connect(database="ebmdb", user="jdqd", password="jdqd", host="139.9.126.19", port="31001")
debug(conn)
cur = conn.cursor()
debug("cur", cur)
cur.execute("select article_id,content_cleared from t_article_msg where (content is not null or content !='' ) order by create_date desc")
rows = cur.fetchall()
for row in rows:
    try:
        print("ID = ", row[0])
        id = row[0]
        content = row[1]
        content = translate_any_2_anyone(content, "en")
        cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print("end---cur_time = ", cur_time)
        content = str(content).replace("'", "\"")
        cur.execute(f"insert into t_article_msg_en(article_id,content) values('{id}','{content}')")
    except Exception as ex:
        conn.rollback()
        if isinstance(ex, UniqueViolation):
            debug(f"Key already exists! article_id={id}")
        else:
            print(f"[ERROR]:[id={id}] ==== ex.type={type(ex)}, ex.msg={ex}")
    conn.commit()

# cur.execute("SELECT * FROM test;")
# cur.fetchone()
cur.close()
conn.close()
