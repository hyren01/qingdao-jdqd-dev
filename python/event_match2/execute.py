from queue import Queue
from subtract_extract import *

main_queue = Queue(maxsize=5)

def get_event_list():

    '''读取事件列表，将（事件ID,事件）元组保存到列表中返回'''
    base_path = os.path.dirname(__file__)
    event_path = base_path+'/allevent'
    try:
        with open(event_path, 'r') as file:
            content = file.read()
    except Exception:
        with open('./allevent', 'r', encoding='utf-8') as file:
            content = file.read()

    event_list = []
    for once in content.split('\n'):
        if once:
            event_list.append(tuple(once.split('`')))

    return event_list


def generate_samples(event_list, summary_sentences):

    '''生成样本供预测模块预测（事件,摘要句子,'0'）'''
    samples = []
    for once in summary_sentences:
        for event in event_list:
            samples.append((event[-1],once,str(0)))

    return samples

def execute(content, title=''):
    '''
    :param title: 文章标题--str
    :param content: 文章内容--str
    :return: event_sorted 事件ID:评分
    '''

    # 获取摘要语句列表
    summary_sentences = get_subtract(content)
    if title:
        summary_sentences.append(title.replace(' ','').strip())

    # 获取事件列表
    event_list = get_event_list()
    # 生成待预测的事件--摘要语句对
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

    if success:
        return event_sorted
    else:
        return pred


if __name__ == '__main__':

    dir_path = './article'
    file_list = os.listdir(dir_path)
    for file_name in file_list:
        file_path = os.path.join(dir_path,file_name)
        title = file_name
        print('\n','*'*20,'开始预测{}'.format(title),'*'*20)

    # file_path = 'D:/work/final_program/article/东亚杯韩国男足首战2比0战胜香港'
    # title = os.path.split(file_path)[-1]
        try:
            with open(file_path, 'r') as file:
                content = file.read()
        except Exception:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()

        print(execute(content, title))
