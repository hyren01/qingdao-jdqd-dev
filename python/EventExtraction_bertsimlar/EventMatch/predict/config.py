# 加载模型中所有的参数

class Config(object):
    '''参数类'''

    def __init__(self):

        # 模型字典
        self.vocab_path = "resources/chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt"

        # 模型config
        self.config_path = "resources/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_config.json"

        # 匹配模型路径
        self.match_model_path = "resources/model/new_best_val_acc_model.h5"

        # ltp模型文件夹路径
        self.ltp_data = "resources/ltp_data"

        # ltp各个模型名称
        # 分词模型
        self.pos_model = "pos.model"

        # 句法解析模型
        self.parser_model = "parser.model"

        # 实体识别模型
        self.ner_model = "ner.model"

        # 语义角色标注模型适用于windows系统
        self.role_model = "pisrl_win.model"
        # 语义角色标注模型适用于linux系统
        # self.role_model = "pisrl.model"

        # ltp加载的字典
        self.ltp_vocab = "resources/ltp_data/vocab.txt"

        # allevent文件路径
        self.allevent_path = "resources/allevent"

        # 字符串编码最大长度
        self.max_length = 256

        # 预测时批量大小
        self.batch_size = 1
