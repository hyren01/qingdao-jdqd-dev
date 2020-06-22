#!/usr/bin/env python
# -*- coding:utf-8 -*-

import json
import helper.event_helper as helper

from flask import Flask, Response, request

app = Flask(__name__)


@app.route('/event_extract', methods=['POST'])
def event_extract():
    """
    事件抽取接口。
    传入参数，如：{"sentence":""}
    返回数据，如：{ "status":"success", "data":[{"subject":"", "verb":"", "object":"", "short_sentence":"",
                    "namedentity":{"person":"", "location":"", "organization":"", "miscellaneous":""},
                    "sentiment_analysis":"", event_datetime":"", "event_location":"", "negative_word":"",
                    "state":"", "event_type":""}] }。
    """
    sentence_txt = request.form.get('sentence')
    event_result = helper.event_extract_helper(sentence_txt)

    return Response(json.dumps({'status': 'success', 'data': event_result}), mimetype="application/json")


@app.route('/constituency_parsed', methods=['POST'])
def constituency_parsed():
    """
    组成成份分析接口。
    传入参数，如：{"sentence":""}
    返回数据，如：{"status":"success", "constituency":"", "content":""}
    """
    sentence_txt = request.form.get('sentence')
    constituency_parse_text = helper.constituency_parsed_helper(sentence_txt)

    return {"status": "success", "constituency": constituency_parse_text, "content": sentence_txt}


@app.route('/coref_with_content', methods=['POST'])
def coref_with_content():
    """
    指代消解接口。
    传入参数，如：{"content":""}
    返回数据，如：{"status":"success", "coref":""}
    """
    try:
        content = request.form.get('content')
        if content is None:
            return {"status": "error", "msg": "content参数为None"}
    except KeyError:
        return {"status": "error"}
    else:
        success, content = helper.coref_with_article_helper(content)
        if success:
            return {"status": "success", "coref": content}
        else:
            return {"status": "error", "coref": content}


if __name__ == '__main__':
    from config.config import Config
    app.run(host='0.0.0.0', port=Config().http_port)
