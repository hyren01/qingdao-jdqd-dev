import obtain_article
import requests
import json

url = "http://172.168.0.115:8083/event_parsed_extract/"
headers = {'content-type': 'application/json'}

if __name__ == '__main__':
    results = obtain_article.get_articles_from_db(2)
    articles = [r[4] for r in results]
    for a in articles:
        requestData = {"content": a}
        ret = requests.post(url, json=requestData, headers=headers)
        if ret.status_code == 200:
            text = json.loads(ret.text)
            print(text)



