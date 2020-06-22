#!/usr/bin/env python
# coding:utf-8
from flask import Flask, request, jsonify
from flask_cors import CORS
from queue import Queue
import threading
import traceback
from feedwork.utils import logger as LOG
from hrconfig import merge_predict_config
from eventmerge.predict import load_model, load_vec, save_vec, delete_vec, data_utils

app = Flask(__name__)
CORS(app)

# 构建参数类
CONFIG = merge_predict_config.Config()
# 加载bert分词器和bert模型
TOKENIZER, BERT_MODEL = load_model.load_bert_model()
# 加载匹配模型
MATCH_MODEL = load_model.load_match_model()

# 匹配用的主队列
MATCH_QUEUE = Queue(maxsize=5)
# 事件向量化保存用的主队列
VEC_QUEUE = Queue(maxsize=5)
# 事件删除主队列
VEC_DELETE_QUEUE = Queue(maxsize=5)
# 向量读取主队列
READ_QUENE = Queue(maxsize=1)


def judge(data):
    """
    传入数据，判断是否为空
    :param data: 数据
    :return: None
    """
    if not data or data is None:
        LOG.error(f"{data} is None!")
        raise ValueError


# 持续循环读取向量文件
def vec_reader():
    """
    遍历读取向量文件，持续循环
    :return: None
    """
    while True:

        message, read_sub_queue = READ_QUENE.get()
        try:
            if message and message is not None:
                # 加载字典文件
                cameo2id = data_utils.read_json(CONFIG.cameo2id_path)
                # 获取所有的cameo号
                cameos = list(cameo2id.keys())
                for cameo in cameos:
                    data = load_vec.load_vec_data(cameo)
                    # 将文件标题以及文章向量内容放入读取子队列中
                    if cameo != cameos[-1]:
                        read_sub_queue.put((True, data))
                    else:
                        read_sub_queue.put((False, data))
        except:
            trace = traceback.format_exc()
            LOG.error(f"read {trace} failed!")

            continue


# 事件相似性匹配
@app.route("/event_match", methods=["GET", "POST"])
def event_match():
    """
    事件匹配，从前端获取事件短句、cameo编号、相似度阈值
    :return: 匹配结果给前端
    :raise:ValueError如果短句为空
    """
    if request.method == "GET":
        # 事件短句，由主语、谓语、否定词、宾语组成
        short_sentence = request.args.get("short_sentence", type=str, default=None)
        # 事件cameo号
        cameo = request.args.get("cameo", type=str, default=None)
        # 相似度阈值
        threshold = request.args.get("threshold", type=float, default=0.5)
    else:
        # 事件短句，由主语、谓语、否定词、宾语组成
        short_sentence = request.form.get("short_sentence", type=str, default=None)
        # 事件cameo号
        cameo = request.form.get("cameo", type=str, default=None)
        # 相似度阈值
        threshold = request.form.get("threshold", type=float, default=0.5)

    if not threshold or threshold is None:

        threshold = 0.5

    # 匹配模块子队列
    match_sub_queue = Queue()
    # 将短句、cameo号、以及阈值传递给匹配模块
    MATCH_QUEUE.put((short_sentence, cameo, threshold, match_sub_queue))
    # 通过匹配模块获取匹配结果
    message, result = match_sub_queue.get()

    if message:
        return jsonify(status="success", result=result)
    else:
        return jsonify(status="failed", result=result)


def match():
    """
    在相应的cameo中查找最相似的事件
    :return: None
    """
    while True:
        short_sentence, cameo, threshold, match_sub_queue = MATCH_QUEUE.get()

        try:
            # 判断短句是否为空
            judge(short_sentence)

            # 短句向量化
            main_vec = load_model.generate_vec(TOKENIZER, BERT_MODEL, short_sentence)
            # 最终的结果
            results = []
            # 判断是否全部遍历
            if not cameo or cameo is None or cameo == "":
                LOG.info("scaning all files...")
                # 文件读取子队列
                read_sub_queue = Queue()
                # 文件读取主队列
                READ_QUENE.put((True, read_sub_queue))

                # 从向量
                while True:
                    status, data = read_sub_queue.get()
                    if status:
                        for once in data:
                            score = load_model.vec_match(main_vec, data[once], MATCH_MODEL)
                            if score >= threshold:
                                results.append({"event_id": once, "score": float(score)})
                    else:
                        for once in data:
                            score = load_model.vec_match(main_vec, data[once], MATCH_MODEL)
                            if score >= threshold:
                                results.append({"event_id": once, "score": float(score)})
                        break
                results.sort(key=lambda x: x["score"], reverse=True)

            else:
                # 加载数据
                data = load_vec.load_vec_data(cameo)
                # 如果data不为空则执行此处操作，否则就任务数据文件为空，这是第一条数据
                if data:
                    for once in data:
                        score = load_model.vec_match(main_vec, data[once], MATCH_MODEL)
                        if score >= threshold:
                            results.append({"event_id": once, "score": float(score)})
                    # 对输出的结果按照降序排序
                    results.sort(key=lambda x: x["score"], reverse=True)

            # 将匹配结果返回给接口
            match_sub_queue.put((True, results))

        except:
            trace = traceback.format_exc()
            LOG.error(trace)
            match_sub_queue.put((False, trace))
            continue


# 事件向量化保存
@app.route("/vec_save", methods=["GET", "POST"])
def event_vectorization():
    """
    事件向量化接口，从前端接收短句、cameo编号、事件id,将事件向量化保存到文件中。
    :return: 保存状态
    :raise:事件短句、事件id为空--ValueError
    """
    if request.method == "GET":
        # 事件短句，由主语、谓语、否定词、宾语组成
        short_sentence = request.args.get("short_sentence", type=str, default=None)
        # 事件cameo号
        cameo = request.args.get("cameo", type=str, default=None)
        # 事件id
        event_id = request.args.get("event_id", type=str, default=None)
    else:
        # 事件短句，由主语、谓语、否定词、宾语组成
        short_sentence = request.form.get("short_sentence", type=str, default=None)
        # 事件cameo号
        cameo = request.form.get("cameo", type=str, default=None)
        # 事件id
        event_id = request.form.get("event_id", type=str, default=None)

    # 向量化模块子队列
    vec_sub_queue = Queue()
    # 将短句、cameo号、以及事件id传递给向量化模块
    VEC_QUEUE.put((short_sentence, cameo, event_id, vec_sub_queue))
    # 获取执行状态
    status, message = vec_sub_queue.get()

    if status:
        return jsonify(status="success", message=message)
    else:
        return jsonify(status="failed", message=message)


def vec_save():
    """
    从前端获取事件短句、cameo号、事件id，将事件短句向量化并保存
    :return: None
    """
    while True:
        # 从接口处获取事件短句、cameo编号、事件id、子队列
        short_sentence, cameo, event_id, vec_sub_queue = VEC_QUEUE.get()
        try:
            # 判断短句是否为空
            judge(short_sentence)
            # 判断cameo是否为空
            judge(cameo)
            # 判断事件id是否为空
            judge(event_id)
            # 事件短句向量化
            main_vec = load_model.generate_vec(TOKENIZER, BERT_MODEL, short_sentence)
            # 向量保存
            save_vec.save_vec_data(cameo, event_id, main_vec)
            # 返回状态值
            vec_sub_queue.put((True, ""))
        except:
            trace = traceback.format_exc()
            LOG.error(trace)
            vec_sub_queue.put((False, trace))
            continue


# 事件向量删除
@app.route("/vec_delete", methods=["GET", "POST"])
def vev_delete():
    """
    事件向量化接口，从前端接收事件id,将事件从事件cameo字典以及npy文件中删除。
    :return: 删除状态
    :raise:事件id为空--ValueError
    """
    if request.method == "GET":
        # 事件id
        event_id = request.args.get("event_id", type=str, default=None)
    else:
        # 事件id
        event_id = request.form.get("event_id", type=str, default=None)

    # 向量化模块子队列
    vec_delete_sub_queue = Queue()
    # 事件id传递给向量删除模块
    VEC_DELETE_QUEUE.put((event_id, vec_delete_sub_queue))
    # 获取执行状态
    status, message = vec_delete_sub_queue.get()

    if status:
        return jsonify(status="success", message=message)
    else:
        return jsonify(status="failed", message=message)


def vec_delete_execute():
    """
    通过队列接收待删除的事件id,遍历cameo_id列表，将id以及对应的事件向量删除。
    :return: None
    """
    while True:
        event_id, vec_delete_sub_queue = VEC_DELETE_QUEUE.get()

        try:
            # 判断短句是否为空
            judge(event_id)
            LOG.info("Begin to delete vector...")
            # 删除事件id以及对应的向量
            state = delete_vec.execute_delete(event_id)

            # 如果state为1则删除成功,0则没有找到对应的event_id
            if state:
                vec_delete_sub_queue.put((True, "success"))
            else:
                vec_delete_sub_queue.put((False, "Event_id not in saved file!"))

        except:
            trace = traceback.format_exc()
            LOG.error(trace)
            vec_delete_sub_queue.put((False, trace))
            continue


if __name__ == "__main__":

    # 向量遍历线程
    t0 = threading.Thread(target=vec_reader)
    t0.daemon = True
    t0.start()
    # 向量匹配线程
    t1 = threading.Thread(target=match)
    t1.daemon = True
    t1.start()
    # 向量保存线程
    t2 = threading.Thread(target=vec_save)
    t2.daemon = True
    t2.start()
    # 向量删除线程
    t3 = threading.Thread(target=vec_delete_execute)
    t3.daemon = True
    t3.start()
    app.run(host="0.0.0.0", port="38083", threaded=True)
