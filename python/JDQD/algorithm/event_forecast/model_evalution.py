import numpy as np
from keras.models import load_model
import time
import sys
import os

module_dir = os.path.dirname(__file__)
import_dir = os.path.join(module_dir, '../..')
log_path = os.path.join(import_dir, 'log/logs')
sys.path.append(import_dir)
from algorithm.event_forecast import predict
from algorithm.event_forecast import preprocess as pp
from algorithm.event_forecast import obtain_data as od
import utils.logger as logger
import utils.pgsql_util as pgsql

LOG = logger.Logger("debug", log_path=log_path)


def aggre_preds(preds):
    """
    将同一事件的多天预测值按照取最大值进行合并
    :param preds: 多个连续样本的预测值
    :return:
    """
    n_samples, n_days, n_classes = preds.shape
    canvas = np.zeros([n_samples, n_samples + n_days - 1, n_classes])
    for i, p in enumerate(preds):
        canvas[i, i: i + n_days] = p
    preds_aggre = np.max(canvas, axis=0)
    return preds_aggre


def to_binary(preds, event_col):
    preds = np.argmax(preds, axis=-1)
    preds = (preds == event_col).astype(int)
    return preds


def evaluate(preds, true_label):
    tp = np.sum(preds * true_label)
    fp = np.sum(preds * (1 - true_label))
    fn = np.sum((1 - preds) * true_label)
    p = tp / (tp + fp) if tp + fp != 0 else 0
    r = tp / (tp + fn)
    f1 = 2 * p * r / (p + r) if tp else 0
    return p, r, f1


def cal_false_report_rate(preds, label_flatten):
    neg_idxes = [i for i, l in enumerate(label_flatten) if l != 1]
    false_pos_idxes = []
    for i, p in enumerate(preds):
        for j, pd in enumerate(p):
            if pd == 0:
                continue
            pos_idx = i + j
            if pos_idx in neg_idxes:
                false_pos_idxes.append(pos_idx)
    num_fp = len(set(false_pos_idxes))
    num_neg = len(neg_idxes)
    return num_fp / num_neg, num_fp, num_neg


def cal_buffered_recall(preds, label_flatten, days_buffer=2):
    pos_idxes = [i for i, l in enumerate(label_flatten) if l == 1]
    pos_idxes_buffered = {p: list(range(p - days_buffer, p + days_buffer + 1)) for p in pos_idxes}
    true_pos_idxes = []
    for i, p in enumerate(preds):
        for j, pd in enumerate(p):
            if pd != 1:
                continue
            pos_idx = i + j
            for pi, pib in pos_idxes_buffered.items():
                if pos_idx in pib:
                    true_pos_idxes.append(pi)
    num_tp = len(set(true_pos_idxes))
    num_pos = len(pos_idxes_buffered)
    return num_tp / num_pos, num_tp, num_pos


def false_alert(preds_flatten, outputs_flatten):
    comb = preds_flatten + outputs_flatten
    comb = (comb > 0).astype(int)
    false = ((preds_flatten - outputs_flatten) == 1).astype(int)
    num_fa = np.sum(false)
    num_comb_pos = np.sum(comb)
    return num_fa / num_comb_pos, num_fa, num_comb_pos


def pred_rank(preds_aggre, label_flatten, top_num, event_col, days_buffer=2):
    """
    对指定事件的预测值进行排序, 计算排名前top_num位中预测正确的比率, 以及预测正确的个数在所有真正例中的比率
    :param top_num:
    :param preds: shape(样本数, 预测天数)
    :param label_flatten: shape(样本数, 1)
    :param days_buffer:
    :return:
    """
    sort_idxes = preds_aggre[:, event_col].argsort()
    top_idxes = sort_idxes[-top_num:]
    true_in_top = 0
    for ti in top_idxes:
        if np.sum(label_flatten[max(0, ti - days_buffer): min(len(label_flatten) - 1, ti + days_buffer)]) > 0:
            true_in_top += 1
    true_positive = len(np.where(label_flatten > 0)[0])
    return true_in_top / top_num, true_in_top / true_positive


def bleu(candidate, reference):
    scores = []
    for i in [1, 2, 3, 4]:
        s_cadi_ap = list()
        s_refer_ap = list()
        s_cadi_dic = dict()
        s_refer_dic = dict()
        gang = 0
        for j in range(0, len(candidate) - i + 1):
            s_cadi = candidate[j:j + i]
            s_cadi_ap.append(str(s_cadi))
            s_refer = reference[j:j + i]
            s_refer_ap.append(str(s_refer))
        for k in s_cadi_ap:
            s_cadi_dic[k] = s_cadi_dic.get(k, 0) + 1

        for k in s_refer_ap:
            s_refer_dic[k] = s_refer_dic.get(k, 0) + 1

        for k in s_cadi_dic.keys():
            if k in s_refer_dic.keys():
                if s_cadi_dic[k] >= s_refer_dic[k]:
                    gang += s_refer_dic[k]
                else:
                    gang += s_cadi_dic[k]

        score = round(gang / len(s_cadi_ap), 2)
        scores.append(score)
    avg_score = (scores[0] + scores[1] + scores[2] + scores[3]) / 4
    return avg_score


def load_models(model_dir):
    encoder = load_model(f"{model_dir}/encoder.h5", compile=False)
    decoder = load_model(f"{model_dir}/decoder.h5", compile=False)
    return encoder, decoder


def load_cnn_model(model_dir):
    return load_model(f"{model_dir}/cnn_model.h5", compile=False)


def pca_decomposition(data):
    data = np.array(data)
    mean = np.mean(data, axis=1)
    diff = data - mean.reshape(len(data), -1)
    cov = 1 / len(data) * diff.T.dot(diff)
    [u, s, v] = np.linalg.svd(cov)
    return u, s, v


def evaluate_sub_model_by_event(event_col, preds, outputs_test):
    preds_aggre = aggre_preds(preds)
    preds = to_binary(preds, event_col)
    preds_shape = preds.shape
    preds_ = np.reshape(preds, [*preds_shape, 1])
    preds_flatten = aggre_preds(preds_)
    preds_flatten = preds_flatten.reshape([-1]).astype(int)
    outputs_test_ = to_binary(outputs_test, event_col)
    if len(outputs_test_.shape) > 1:
        outputs_test_ = pp.flatten_outputs(outputs_test_)
    if len(preds_flatten) > len(outputs_test_):
        preds_flatten = preds_flatten[:len(outputs_test_)]
    eval_event = {}
    tier_precision, tier_recall = pred_rank(preds_aggre, outputs_test_, 10, event_col)
    bleu_score = bleu(preds_flatten, outputs_test_)
    fr, num_fp, num_neg = cal_false_report_rate(preds, outputs_test_)
    r2, num_tp, num_pos_rc = cal_buffered_recall(preds, outputs_test_)
    fa, num_fa, num_comb_pos = false_alert(preds_flatten, outputs_test_)
    eval_event['rank'] = [tier_precision, tier_recall]
    eval_event['bleu'] = bleu_score
    eval_event['fr'] = fr
    eval_event['num_fp'] = num_fp
    eval_event['num_neg'] = num_neg
    eval_event['rc'] = r2
    eval_event['num_tp'] = num_tp
    eval_event['num_pos_rc'] = num_pos_rc
    eval_event['fa'] = fa
    eval_event['num_fa'] = num_fa
    eval_event['num_comb_pos'] = num_comb_pos
    return eval_event


def evaluate_sub_models(data, dates, detail_ids, sub_model_dirs, params_list, events_p_oh, events_set, n_classes, start_date,
                        end_date, w1, w2, w3, w4):
    scores = []
    events_num = {}

    for detail_id, sub_model_dir, params in zip(detail_ids, sub_model_dirs, params_list):
        LOG.info(f'评估模型: {sub_model_dir}, detail_id: {detail_id}')
        input_len, output_len, n_pca = params
        preds, outputs_test = pred(sub_model_dir, data, dates, events_p_oh, input_len, output_len, n_classes,
                                   n_pca, start_date, end_date)
        events_num = pp.get_event_num(outputs_test, events_set)
        bleus = []
        tier_precisions = []
        tier_recalls = []
        num_fps = []
        num_negs = []
        num_tps = []
        num_pos_rcs = []
        num_fas = []
        num_comb_poses = []

        for i, event in enumerate(events_set):
            if str(event) == '0':
                continue
            event_num = events_num[event]
            if event_num == 0:
                continue
                # @todo 有的指标没有事件也是可以计算的, 如误报率, 需增加
            evals = evaluate_sub_model_by_event(i, preds, outputs_test)

            false_rate = round(evals['fr'], 4)
            num_fp = evals['num_fp']
            num_neg = evals['num_neg']
            num_fps.append(num_fp)
            num_negs.append(num_neg)

            recall_rate = round(evals['rc'], 4)
            num_tp = evals['num_tp']
            num_pos_rc = evals['num_pos_rc']
            num_tps.append(num_tp)
            num_pos_rcs.append(num_pos_rc)

            false_alarm_rate = round(evals['fa'], 4)
            num_fa = evals['num_fa']
            num_comb_pos = evals['num_comb_pos']
            num_fas.append(num_fa)
            num_comb_poses.append(num_comb_pos)

            tier_precision, tier_recall = evals['rank']
            tier_precision = round(tier_precision, 4)
            tier_recall = round(tier_recall, 4)
            tier_precisions.append(tier_precision)
            tier_recalls.append(tier_recall)

            bleu = round(evals['bleu'], 4)
            bleus.append(bleu)

            pgsql.insert_model_test(event, event_num, false_rate, recall_rate, false_alarm_rate, tier_precision,
                                    tier_recall, bleu, detail_id)

        if bleus:
            bleu_summary = round(np.mean(bleus), 4)
            tier_precision_summary = round(np.mean(tier_precisions), 4)
            tier_recall_summary = round(np.mean(tier_recalls), 4)
            fr_summary = round(np.sum(num_fps) / np.sum(num_negs), 4)
            rc_summary = round(np.sum(num_tps) / np.sum(num_pos_rcs), 4)
            fa_summary = round(np.sum(num_fas) / np.sum(num_comb_poses), 4)
            score = w1 * bleu_summary + w2 * rc_summary + w3 * (1 - fr_summary) + w4 * (1 - fa_summary)
            score = round(score, 4)
            scores.append([score, bleu_summary, tier_precision_summary, tier_recall_summary, fr_summary, rc_summary,
                           fa_summary, detail_id])

    LOG.info('模型评估结束, 筛选top模型')
    pgsql.insert_model_tot(scores, events_num)


def pred(model_dir, data, dates, events_p_oh, input_len, output_len, n_classes, n_pca, start_date, end_date):
    models_dir = f'{module_dir}/resources/models'
    values_pca = pp.apply_pca(n_pca, models_dir, data, True)

    inputs_test, outputs_test = pp.gen_samples_by_date(values_pca, events_p_oh, input_len, output_len, dates,
                                                       start_date, end_date)

    preds = predict.predict(model_dir, inputs_test, output_len, n_classes)
    preds = np.array(preds)
    return preds, outputs_test


def format_time(timestamp):
    formatted = time.localtime(timestamp)
    return time.strftime('%Y-%m-%d %H:%M:%S', formatted)


def get_file_create_time(filepath):
    t = os.path.getctime(filepath)
    return format_time(t)


if __name__ == '__main__':
    import os

    dates, data = od.combine_data(['data_xlshuju_1', 'data_xlshuju_2', 'data_xlshuju_3'], None, True)
