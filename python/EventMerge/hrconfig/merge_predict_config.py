# coding:utf-8
# 用于传输代码中的所有参数以及路径


class Config(object):
    """
    构建参数类，向代码中传递参数
    """
    def __init__(self,):
        # bert模型的参数
        self.bert_config = "eventmerge/resources/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_config.json"
        # bert模型ckpt文件
        self.checkpoint_path = "eventmerge/resources/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_model.ckpt"
        # bert模型使用给定字典
        self.vocab_path = "eventmerge/resources/chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt"
        # bert所能编码的最大长度
        self.maxlen = 512
        # 匹配模型路径
        self.match_model_path = "eventmerge/resources/model/match_model.h5"
        # 向量保存的文件夹
        self.vec_data_dir = "eventmerge/resources/vec_data"
        # cameo:id 字典文件
        self.cameo2id_path = "eventmerge/resources/cameo2id.json"
