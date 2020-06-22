import obtain_article
import requests
import json

url = "http://172.168.0.115:8083/event_parsed_extract/"
headers = {'content-type': 'application/json'}


def parse_article(article):
    requestData = {"content": article}
    ret = requests.post(url, json=requestData, headers=headers)
    if ret.status_code == 200:
        text = json.loads(ret.text)
        print(text)


if __name__ == '__main__':
    s = '如果他明天过来, 就告诉他我的杯子打碎了. '
    r = parse_article(s)
    results = obtain_article.get_articles_from_db(2)
    articles = [r[4] for r in results]
    for a in articles:
        parse_article(a)
