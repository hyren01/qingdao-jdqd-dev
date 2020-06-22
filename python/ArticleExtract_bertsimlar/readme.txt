numpy
tensorflow==1.13.2
keras==2.3.1


该模块主要对传入query--str,进行检索

    flask_main.py为模型web接口程序

            接收:待检索query--str,

            输出：[{'title':title， 'score':score}, ]

    resources ：

        allfile ： 存储所有的文章

        model ：
            new_best_val_acc_model.h5  模型文件
            vocab.txt 模型调用的字典文件

    predict ：

        BertSimlar 用keras实现的bert主程序

        data_util.py 对文本文件进行清洗

        execute.py 调用data_util生成样本

        logger 日志模块


