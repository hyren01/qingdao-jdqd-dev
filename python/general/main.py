import obtain_article
import requests
import json
import os

url = "http://172.168.0.115:8083/event_parsed_extract"

status_codes = []


def parse_article(article, id):
    requestData = {"content": article, "content_id": id}
    ret = requests.post(url, data=requestData)
    status_code = ret.status_code
    status_codes.append(status_code)
    if status_code == 200:
        text = json.loads(ret.text)
        print(text)


if __name__ == '__main__':
    results = obtain_article.get_articles_from_db()
    article_ids = [[r[1], r[0]] for r in results]
    record_file = 'records.txt'
    if not os.path.exists(record_file):
        records = []
        processed_ids = []
    else:
        with open(record_file, 'r', encoding='utf-8') as f:
            records = f.readlines()
        records = [r.replace('\n', '').split(',') for r in records]
        processed_ids = [r[0] for r in records]
    with open(record_file, 'a', encoding='utf-8') as f:
        for a, id in article_ids:
            if id in processed_ids:
                continue
            print('parsing article', id)
            data = {"content": a}
            try:
                parse_article(a, id)
                record = id + ',' + 's\n'
            except:
                record = id + ',' + 'f\n'
            finally:
                f.write(record)
