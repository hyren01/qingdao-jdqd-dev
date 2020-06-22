# 模型训练
    
    - 存放模型训练相关的内容
    
    - 使用方法：
        
        1、按照requiremens安装训练需要的包
        
        2、将对应的训练数据放置到相应的文件夹下
        
        3、执行.py文件即可或者调用flask_main.py进行训练
        
    接口：127.0.0.1:38082/event_train
    
    输入：{"train_type": "extract"}  extract  cameo  state
    
    输出：{"status": "success"}

## model

    - initial_model 存放bert模型初始化ckpt文件及字典
    
    - event_cameo_trained_model 存放训练后的cameo模型
    
    - event_extract_trained_model 存放事件抽取训练后的模型
      
    - event_state_trained_model 存放事件状态训练后的模型
      
## resources
     
     - event_cameo_data 存放事件cameo训练的数据
     
     - event_extract_data 存放事件抽取的训练数据
     
        1、raw_data 原始训练数据
        
        2、supplement 补充训练数据
        
     - event_state_data 事件状态训练数据

## utils
    
    - event_cameo_data_util.py 事件cameo数据处理模块
    
    - event_extract_data_util.py 事件抽取数据处理模块
    
    - event_state_data_util.py 事件状态数据处理模块
    
    - utils.py 所有模块用的数据生成以及字典处理模块

## event_cameo_train.py

    - 事件cameo训练模块

## event_extract_train.py

    - 事件论元抽取训练模块
    
## event_state_train.py

    - 事件状态训练模块
    