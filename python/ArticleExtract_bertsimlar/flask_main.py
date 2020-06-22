# coding: utf-8
import os
import numpy as np
from predict.BertSimlar.utils import ArgConfig, data_generator, print_arguments, init_log, load_data
from bert4keras.tokenizer import Tokenizer
from bert4keras.backend import keras, set_gelu, K
from bert4keras.snippets import sequence_padding, get_all_attributes
from bert4keras.bert import build_bert_model
from bert4keras.optimizers import Adam
from predict import execute
from predict import data_util
from predict import logger

LOG = logger.Logger('info')

locals().update(get_all_attributes(keras.layers))  # from keras.layers import *

from keras.models import load_model
from flask import Flask, request, jsonify
from flask_cors import CORS
from queue import Queue
import traceback
import threading


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
    '''
    从前端获取待检索的短句--str
    :return: 返回预测结果
    '''

    if request.method == "POST":
        query = request.form.get("query", type=str, default=None)
    else:
        query = request.args.get("query", type=str, default=None)


    sub_queue = Queue()
    main_queue.put((query, sub_queue))
    success, pred= sub_queue.get()


    if success:
        return successResp(pred)
    else:
        return errorResp(pred)


def get_title2id(title_path="./resources/title_list.txt"):
    '''
    读取title_list.txt文本，创建id2title字典
    :param title_path:title_list文件目录
    :return:返回id2title字典--dict
    '''
    content = execute.file_reader(title_path)
    title_list = content.split('\n')
    title2id = {once.split('\t')[1][1:-1].replace('.txt', '') : once.split('\t')[0][1:-1] for once in title_list if once}

    return title2id


def worker(file_dir = './resources/allfile'):
    '''
    传入文章存储的文件夹路径，对传入的query进行检索
    :param file_dir: 保存文件的文件夹路径
    :return: None
    '''
    maxlen = 256
    batch_size = 1
    vocab_path = './resources/model/vocab.txt'
    model_path = './resources/model/new_best_val_acc_model.h5'

    tokenizer = Tokenizer(vocab_path)
    model = load_model(model_path, compile=False)
    # 加载标题--id字典
    title2id = get_title2id(title_path="./resources/title_list.txt")

    def evaluate(data):
        results = []
        for x_true, y_true in data:
            y_pred = model.predict(x_true)
            results += np.reshape(y_pred[:, 1], (-1,)).tolist()
        return results

    while True:

        # 获取数据和子队列
        query, sub_queue = main_queue.get()

        try:
            # 获取文件列表读取文件内容
            file_list = os.listdir(file_dir)
            results = {}
            i = 0
            for file in file_list:
                i +=1
                LOG.info("检索文章 {}".format(i))           	
                file_path = os.path.join(file_dir, file)
                content = execute.file_reader(file_path)
                content = data_util.process_data(content)
                sentences = execute.cut_sentence(content)
                sentences.append(file.replace('.txt', ''))
                samples = execute.generate_samples(query, sentences)
                data_generator_ = data_generator(samples, tokenizer, max_length=maxlen, batch_size=batch_size)
                pred = evaluate(data_generator_)
                title = file.replace('.txt', '')
                results[title2id[title]] = max(pred)

            # 将预测得到的结果进行排序，放入队列中传递给前端，[{'article_id':str,'score':float }]
            results = execute.sort_socres(results)
            LOG.info("返回检索结果。") 
            sub_queue.put((True, results))

        except:
            # 通过子队列发送异常信息
            trace = traceback.format_exc()
            LOG.info(trace)
            sub_queue.put((False, trace))

            continue



if __name__ == '__main__':

    t = threading.Thread(target=worker)
    t.daemon = True
    t.start()
    app.run(host='0.0.0.0', port=38083, threaded=True)
