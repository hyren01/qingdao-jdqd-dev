# 该模块功能

    - 端口：38081
    
    - 本机调试：127.0.0.1:38081/ematch?title=日本派遣宙斯盾军舰赶赴东部海域.&content=
    
    - 对输入的中文文章标题以及内容进行事件匹配，输出匹配结果
    
    - 输入:{"title":"",
    
            "content":"",
            
            "sample_type:""(也可以不用传，底层默认parts)[parts abstract triples]
            
            }
    
    
    -输出: {"code": 0,
    
            "data":[{"title_pred":[
                    
                        {"event_id": "3",
                        
                         "ratio": 0.753}]},
                         
                     {"content_pred":[
                        
                        {"event_id": "3",
                        
                         "ratio": 0.753}]}],
                     
            "message": "success"}
            
## 使用前准备

    将对应的模型存放到对应的文件夹下，然后运行main.py文件即可

## predict

    模型预测需要使用的所有代码。
    
    1、config.py 保存代码中的路径以及各种参数类
    
    2、data_util.py  对输入的字符串数据进行清洗
    
    3、execute.py  功能主体模块，加载模型，预测，并对预测结果进行处理返回给main.py
    
    4、logger.py  日志模块
    
    5、get_abstract.py 摘要模块
    
    6、get_triples.py 三元组抽取模块
    


## resourses
    
    1、bert4keras_v01--存放bert模型结构代码
    
    2、chinese_roberta_wwm_ext_L-12_H-768_A-12--存放bert模型的参数以及字典
    
    3、model--存放训练好的模型文件
    
    4、ltp_data--存放pyltp需要的模型以及字典
    
    5、allevent--事件列表文件--格式： 66`朝内部权力  变化
    
## train
    
    1、model--initial_model存放bert初始化参数及模型
    
              trained_model存放训练后的模型
              
    2、resourses--存放训练数据以及第三方包
    
    3、logger.py--日志模块
    
    4、train_similarity_model.py--事件匹配模型首次训练
    
       train_similarity_model.sh--事件匹配模型首次训练shell文件
    
    5、second_train.py--事件匹配模型二次训练
    
       second_train.sh--事件匹配模型二次选了shell文件

## restapp 

    main.py--flask主控接口