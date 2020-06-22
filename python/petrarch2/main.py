#!/usr/bin/env python
# -*- coding:utf-8 -*-

from flask import Flask, request
from config.config import Config
from services.petrarch2 import run_by_constituency
from helper.init_dictionary import init_params

app = Flask(__name__)


@app.route('/get_event_code', methods=['POST'])
def get_event_code():
    """
    事件类型分析接口。
    :return: json数据，如：{'status': 'success', 'cameo_code': ''}
    """
    constituency_txt = request.form.get('constituency')
    content = request.form.get('content')
    cameo_code = run_by_constituency(content, constituency_txt)

    return {'status': 'success', 'cameo_code': cameo_code}


if __name__ == '__main__':
    init_params()
    app.run(host='0.0.0.0', port=Config().http_port)
