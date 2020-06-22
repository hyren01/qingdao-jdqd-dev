# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 11:02:38 2020

@author: 12894
"""

from flask import Flask, request
import json

app = Flask(__name__)

app.config.update(RESTFUL_JSON=dict(ensure_ascii=False))

import Model.predict as tri_pred

model = tri_pred.load_model('resc/best_model.h5')

@app.route("/relation_keywords", methods=['GET', 'POST'])
def extract_keywords():
    sentence = request.form.get('sentence')

    tag = tri_pred.extract_items(model, sentence)

    single_pr, causes_pr, ends_pr = '', '', ''
    for s, t in zip(sentence, tag):
        if t in ('B-S', 'I-S'):
            single_pr += ' ' + s if (t == 'B-S') else s
        if t in ('B-C', 'I-C'):
            causes_pr += ' ' + s if (t == 'B-C') else s
        if t in ('B-E', 'I-E'):
            ends_pr += ' ' + s if (t == 'B-E') else s
    single_pr = list(set(single_pr.split()))
    causes_pr = list(set(causes_pr.split()))
    ends_pr = list(set(ends_pr.split()))
    rst = {'single': single_pr, 'multi1': causes_pr, 'multi2': ends_pr}
    return json.dumps(rst)


if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=12319)
