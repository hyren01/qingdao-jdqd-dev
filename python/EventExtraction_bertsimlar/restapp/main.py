# coding:utf-8

import os
import sys

file = os.path.abspath(__file__)
dir_path = os.path.dirname(file)
dir_path = os.path.join(dir_path, "../")
sys.path.append(dir_path)

from flask import Flask, request, jsonify
from flask_cors import CORS
from queue import Queue
import traceback
import threading

app = Flask(__name__)
CORS(app)

main_queue = Queue(maxsize=5)

# 请求失败
def errorResp(msg):
    '''
    将请求失败后的信息json化
    :param msg: 请求失败信息
    :return: json
    '''
    return jsonify(code=-1, message=msg)


# 成功请求
def successResp(data):
    '''
    将预测数据json化
    :param data: 事件相似度
    :return: json化的数据
    '''
    return jsonify(code=0, message="success", data=data)


@app.route('/ematch', methods=['GET', 'POST'])
def infer():
    '''
    从前端接受文章标题、内容、样本类型
    :return: 事件相似度列表
    '''
    if request.method == "POST":
        title = request.form.get("title", type=str, default=None)
        content = request.form.get("content", type=str, default=None)
        sample_type = request.form.get("sample_type", type=str, default='parts')
    else:
        title = request.args.get("title", type=str, default=None)
        content = request.args.get("content", type=str, default=None)
        sample_type = request.args.get("sample_type", type=str, default='parts')

    if not content:
        return errorResp('文章内容为空')
    elif not title:
        return errorResp('文章标题为空')

    # 创建子队列，从预测模块获取处理信息
    sub_queue = Queue()
    # 使用主队列将请求内容以及子队列传入预测模块
    main_queue.put((title, content, sample_type, sub_queue))
    # 使用子队列从预测模块获取请求信息以及预测数据
    success, pred = sub_queue.get()

    if success:
        return successResp([{'title_pred': pred[0]}, {'content_pred': pred[1]}])
    else:
        return errorResp(pred)


def worker():
    '''
    预测模块，对主队列传入的文本进行处理及事件匹配预测，通过子队列将结果返回
    :return: None
    '''
    from EventMatch.predict.execute import load_match_model, get_events, get_predict_result
    # 加载事件匹配模型
    model = load_match_model()
    # 获取事件列表
    event_list = get_events()


    while True:

        # 获取数据和子队列
        title, content, sample_type, sub_queue = main_queue.get()
        try:

            title_pred, content_pred = get_predict_result(model, event_list, title, content, sample_type)

            sub_queue.put((True, [title_pred, content_pred]))

        except:
            # 通过子队列发送异常信息
            trace = traceback.format_exc()
            sub_queue.put((False, trace))

            continue


if __name__ == '__main__':
    t = threading.Thread(target=worker)
    t.daemon = True
    t.start()
    app.run(host='0.0.0.0', port=38081, threaded=True)
