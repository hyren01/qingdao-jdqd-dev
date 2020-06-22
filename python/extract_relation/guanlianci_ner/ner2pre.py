# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 11:02:38 2020

@author: 12894
"""

import Model.ner_predict as tri_pred

sentence = '因为罗兴亚人问题和事件本身可以说涉及到人权问题，所以美国会根据美国自己在这方面的想法或者制度来制定相关的政策'   
tag  = tri_pred.extract_items(sentence)

single_pr, causes_pr, ends_pr = '', '', ''
for s, t in zip(sentence, tag):
    if t in ('B-S', 'I-S'):
        single_pr += ' ' + s if (t == 'B-S') else s
    if t in ('B-C', 'I-C'):
        causes_pr += ' ' + s if (t == 'B-C') else s
    if t in ('B-E', 'I-E'):
        ends_pr += ' ' + s if (t == 'B-E') else s
single_pr = set(single_pr.split())
causes_pr = set(causes_pr.split())
ends_pr = set(ends_pr.split())
  
print(f'sentence: {sentence}' + '\n' + f'single: {single_pr}' + '\t' + f'causes_pr {causes_pr}' + '\t' + f'ends_pr: {ends_pr}')