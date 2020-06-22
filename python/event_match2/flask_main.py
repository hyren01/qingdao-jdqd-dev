import os
from execute import get_event_list, generate_samples
from subtract_extract import get_subtract
from keras.models import load_model
from BertSimlar.utils import data_generator
from flask import Flask, request, jsonify
from flask_cors import CORS
from queue import Queue
import traceback
import threading
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from bert4keras.tokenizer import Tokenizer
from bert4keras.backend import keras, set_gelu, K
from bert4keras.bert import build_bert_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, get_all_attributes

locals().update(get_all_attributes(keras.layers))  # from keras.layers import *


app = Flask(__name__)
CORS(app)

main_queue = Queue(maxsize=5)


def errorResp(msg):
    return jsonify(code=-1, message=msg)


# 成功请求
def successResp(data):
    return jsonify(code=0, message="success", data=data)


@app.route('/ematch', methods=['GET', 'POST'])
def infer():
    if request.method == "POST":
        title = request.form.get("title", type=str, default=None)
        content = request.form.get("content", type=str, default=None)
    else:
        title = request.args.get("title", type=str, default=None)
        content = request.args.get("content", type=str, default=None)
    if not content:
        return errorResp('文章内容为空')

    
    summary_sentences = get_subtract(content)
    if title:
        summary_sentences.append(title.replace(' ', '').strip())
    
    event_list = get_event_list()
    
    samples = generate_samples(event_list, summary_sentences)

    sub_queue = Queue()
    main_queue.put((samples, sub_queue))
    success, pred = sub_queue.get()

    predicted_event = {}
    event_scores = {}
    # event_sorted = []
    for key in event_list:
        predicted_event[key[1]] = [key[0], []]
    for once in pred:
        predicted_event[once[0]][1].append(once[-1])
    for i in predicted_event:
        event_scores[predicted_event[i][0]] = max(predicted_event[i][1])
    event_sorted = list(sorted(event_scores.items(), key=lambda e: e[1], reverse=True))
    event_sorted = [{'event_id': elem[0], 'ratio': elem[1]} for elem in event_sorted]
    # for once in event_output:
    #     event_sorted.append((once[0],once[1]))

    if success:
        return successResp(event_sorted)
    else:
        return errorResp(pred)


def worker():
    maxlen = 256
    batch_size = 1

    base_path = os.path.abspath('.')
    vocab_path = base_path + '/BertSimlar/model/vocab.txt'
    model_path = base_path + '/BertSimlar/model/new_best_val_acc_model.h5'

    tokenizer = Tokenizer(vocab_path)
    model = load_model(model_path, compile=False)

    def evaluate(data):
        results = []
        for x_true, y_true in data:
            y_pred = model.predict(x_true)
            results += np.reshape(y_pred[:, 1], (-1,)).tolist()
        return results

    while True:
        
        data, sub_queue = main_queue.get()
        try:
            
            test_generator = data_generator(data, tokenizer, max_length=maxlen, batch_size=batch_size)
            results = evaluate(test_generator)

            predicted_event = []
            for elem, pred in zip(test_generator.data, results):
                predicted_event.append(list(elem[0:2]) + [pred])

            sub_queue.put((True, predicted_event))
        except:
            
            trace = traceback.format_exc()
            print(trace)
            sub_queue.put((False, trace))
            continue


if __name__ == '__main__':
    t = threading.Thread(target=worker)
    t.daemon = True
    t.start()
    app.run(host='0.0.0.0', port=38081, threaded=True)
