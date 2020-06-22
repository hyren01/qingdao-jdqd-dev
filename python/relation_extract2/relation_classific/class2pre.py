# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 14:35:58 2020

@author: 12894
"""

from flask import Flask, request
import json

import class_predict as pre

app = Flask(__name__)

app.config.update(RESTFUL_JSON=dict(ensure_ascii=False))

model = pre.load_model('resc/best_causality_class_model.h5')

@app.route("/relation_classify", methods=['GET', 'POST'])
def split():
    event1 = request.form.get('event1')
    event2 = request.form.get('event2')
    prob = pre.class_pre(model, event1, event2)
    return json.dumps({'type': int(prob)})


if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=12318)
