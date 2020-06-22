from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import *
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_accuracy
from keras.preprocessing.sequence import pad_sequences
from keras_bert import load_trained_model_from_checkpoint
from keras_bert import Tokenizer
from Model import DealWithData
import keras.backend as K

label = {}
_label = {}
max_seq_length = 160
batch_size = 12
epochs = 2
lstmDim = 64
learning_rate = 5e-5
min_learning_rate = 1e-5

config_path = './chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = './chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = './chinese_L-12_H-768_A-12/vocab.txt'
label_path = './Parameter/tag_dict.txt'

f_label = open(label_path, 'r+', encoding='utf-8')
for line in f_label:
    content = line.strip().split()
    label[content[0].strip()] = content[1].strip()
    _label[content[1].strip()] = content[0].strip()
#dict
vocab = {}
with open(dict_path, 'r+', encoding='utf-8') as f_vocab:
    for line in f_vocab.readlines():
        vocab[line.strip()] = len(vocab)
        
       
    
train_path = "./data/contrast/train_contrast_line.txt"
test_path = "./data/contrast/test_contrast_line.txt"
input_train, result_train = DealWithData.PreProcessData(train_path)
input_test, result_test = DealWithData.PreProcessData(test_path) 

#预处理输入数据
def PreProcessInputData(text):
    tokenizer = Tokenizer(vocab)
    word_labels = []
    seq_types = []
    for sequence in text:
        code = tokenizer.encode(first=sequence, max_len=max_seq_length)
        word_labels.append(code[0])
        seq_types.append(code[1])
    return word_labels, seq_types

#预处理结果数据
def PreProcessOutputData(text):
    tags = []
    for line in text:
        tag = [0]
        for item in line:
            tag.append(int(label[item.strip()]))
        tag.append(0)
        tags.append(tag)

    pad_tags = pad_sequences(tags, maxlen=max_seq_length, padding="post", truncating="post")
    result_tags = np.expand_dims(pad_tags, 2)
    return result_tags
    
#训练集
input_train_labels, input_train_types = PreProcessInputData(input_train)
result_train_pro = PreProcessOutputData(result_train)
#测试集
input_test_labels, input_test_types = PreProcessInputData(input_test)
result_test_pro = PreProcessOutputData(result_test)   
    
bert = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=max_seq_length)

#make bert layer trainable
for layer in bert.layers:
    layer.trainable = True
    
x1 = Input(shape=(None,))
x2 = Input(shape=(None,))
bert_out = bert([x1, x2])
#lstm_out = Bidirectional(LSTM(lstmDim,
#                                 return_sequences=True,
#                                 dropout=0.2,
#                                 recurrent_dropout=0.2))(bert_out)
crf_out = CRF(len(label), sparse_target=True)(bert_out)
model = Model([x1, x2], crf_out)
model.summary()
model.compile(
    optimizer=Adam(1e-4),
    loss=crf_loss,
    metrics=[crf_accuracy]
)


def Id2Label(ids):
    result = []
    for id in ids:
        result.append(_label[str(id)])
    return result

def Vector2Id(tags):
    result = []
    for tag in tags:
        result.append(np.argmax(tag))
    return result

def extract_items(sentence):
    sentence = sentence[:max_seq_length-1]
    labels, types = PreProcessInputData([sentence])
    tags = model.predict([labels, types])[0]
    result = []
    for i in range(1, len(sentence) + 1):
        result.append(tags[i])
    result = Vector2Id(result)
    tag = Id2Label(result)
    return tag
    
class Evaluate(Callback):
    def __init__(self):
        self.F1 = []
        self.best = 0.
        self.passed = 0
        self.stage = 0

    def on_epoch_end(self, epoch, logs=None):
        f1, precision, recall = self.evaluate()
        self.F1.append(f1)
        if f1 > self.best:
            self.best = f1
            model.save('best_model.h5', include_optimizer=True)
        print('epoch: %d, f1: %.4f, precision: %.4f, recall: %.4f, best f1: %.4f\n' % (epoch, f1, precision, recall, self.best))

    @staticmethod
    def flat_lists(lists):
        all_elements = []
        for lt in lists:
            all_elements.extend(lt)
        return all_elements

    def evaluate(self):
        single_A, single_B, single_C = 1e-10, 1e-10, 1e-10
        causes_A, causes_B, causes_C = 1e-10, 1e-10, 1e-10
        ends_A, ends_B, ends_C = 1e-10, 1e-10, 1e-10
        for i in range(len(input_test)):
            input_line = input_test[i]
            result_line = result_test[i]
            tag = extract_items(input_line)
            single_tr, causes_tr, ends_tr = '', '', ''
            single_pr, causes_pr, ends_pr = '', '', ''
            for s, t in zip(input_line, result_line):
                if t in ('B-S', 'I-S'):
                    single_tr += ' ' + s if (t == 'B-S') else s
                if t in ('B-C', 'I-C'):
                    causes_tr += ' ' + s if (t == 'B-C') else s
                if t in ('B-E', 'I-E'):
                    ends_tr += ' ' + s if (t == 'B-E') else s
            single_tru = set(single_tr.split())
            causes_tru = set(causes_tr.split())
            ends_tru = set(ends_tr.split())
            for s, t in zip(input_line, tag):
                if t in ('B-S' ,'I-S'):
                    single_pr += ' ' + s if (t == 'B-S') else s
                if t in ('B-C', 'I-C'):
                    causes_pr += ' ' + s if (t == 'B-C') else s
                if t in ('B-E', 'I-E'):
                    ends_pr += ' ' + s if (t == 'B-E') else s
            single_pre = set(single_pr.split())
            causes_pre = set(causes_pr.split())
            ends_pre = set(ends_pr.split())
            
            single_A += len(single_tru & single_pre)
            single_B += len(single_pre)
            single_C += len(single_tru)
            
            causes_A += len(causes_tru & causes_pre)
            causes_B += len(causes_pre)
            causes_C += len(causes_tru)
            ########
            ends_A += len(ends_tru & ends_pre)
            ends_B += len(ends_pre)
            ends_C += len(ends_tru)
            
        single_f1, single_precision, single_recall = 2 * single_A / (single_B + single_C), \
                                                        single_A / single_B, single_A / single_C
                                                        
        causes_f1, causes_precision, causes_recall = 2 * causes_A / (causes_B + causes_C), \
                                                        causes_A / causes_B, causes_A / causes_C
        
        ends_f1, ends_precision, ends_recall = 2 * ends_A / (ends_B + ends_C), \
                                                                    ends_A / ends_B, ends_A / ends_C
        
        f1 = (single_f1 + causes_f1 + ends_f1) / 3
        precision = (single_precision + causes_precision + ends_precision) / 3
        recall = (single_recall + causes_recall + ends_recall) / 3

        return f1, precision, recall
        
def model_test(input_test):
    """输出测试结果
    """
    with open ('test_line_pre.txt', 'w', encoding = 'utf-8') as f:
        for i in range(len(input_test)):
            input_line = input_test[i]
            result_line = result_test[i]
            print(input_line)
            tag = extract_items(input_line)
            single_tr, causes_tr, ends_tr = '', '', ''
            single_pr, causes_pr, ends_pr = '', '', ''
            for s, t in zip(input_line, result_line):
                if t in ('B-S', 'I-S'):
                    single_tr += ' ' + s if (t == 'B-S') else s
                if t in ('B-C', 'I-C'):
                    causes_tr += ' ' + s if (t == 'B-C') else s
                if t in ('B-E', 'I-E'):
                    ends_tr += ' ' + s if (t == 'B-E') else s
            single_tru = set(single_tr.split())
            causes_tru = set(causes_tr.split())
            ends_tru = set(ends_tr.split())
            for s, t in zip(input_line, tag):
                if t in ('B-S', 'I-S'):
                    single_pr += ' ' + s if (t == 'B-S') else s
                if t in ('B-C', 'I-C'):
                    causes_pr += ' ' + s if (t == 'B-C') else s
                if t in ('B-E', 'I-E'):
                    ends_pr += ' ' + s if (t == 'B-E') else s
            single_pru = set(single_pr.split())
            causes_pru = set(causes_pr.split())
            ends_pru = set(ends_pr.split())
            f.write('sentence: ' + input_line + '\n' + 'single_tru: ' + str(single_tru) + '\t' + 'causes_tru: ' + str(causes_tru) + '\t' + 
                        'ends_tru: ' + str(ends_tru) + '\n' + 'single_pru: ' + str(single_pru) + '\t' + 'causes_pru: ' + str(causes_pru) + '\t' + 
                        'ends_pru: ' + str(ends_pru) + '\n\n')
        
evaluator = Evaluate()

if __name__ == '__main__':
    history = model.fit(x=[input_train_labels, input_train_types],
                   y=result_train_pro,
                   batch_size=batch_size,
                   epochs=epochs,
                   validation_data=[[input_test_labels, input_test_types], result_test_pro],
                   verbose=1,
                   shuffle=True,
                   callbacks=[evaluator]
                   )

    model.load_weights("best_model.h5")
    model_test(input_test)
    f1, precision, recall = evaluator.evaluate()
    print("f1:{}, precision:{}, recall:{}".format(f1, precision, recall))




