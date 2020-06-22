程序启动方法:
1. 将JDQD目录及其下面所有文件放到/opt/hyren/forecast/目录下
1. 执行 source opt/tf14/bin/activate 激活Python环境
2. 执行 python /opt/hyren/forecast/JDQD/services/main.py 启动算法服务

参数说明:
1. services/main.py(19-27行):
    con = 'postgresql': 数据库连接方式, 根据所连接的产品不同, 可换成'gbase', 'mysql'
    w1 = 0.1: 整合评估指标时bleu score的权重
    w2 = 0.5: 整合评估指标时召回率的权重
    w3 = 0.3: 整合评估指标时误报率的权重
    w4 = 0.1: 整合评估指标时虚警率的权重
    super_event_col = 1: 大类事件所在事件表的列下标(从0开始)
    sub_event_col = 2: : 小类事件所在事件表的列下标(从0开始)
    date_col = 5: 日期列所在数据表的列下标(从0开始)
    event_priority = 11209: 遇到一天发生多个事件时优先保留的事件名称

2. config/dbconn_cfg:
    模型训练与预测的数据表所在数据库连接配置的键值对, 需通过修改值更改配置
    DATABASE: 数据库名称
    DBUSER: 用户名
    PASSWORD: 密码
    HOST: 数据库主机地址
    PORT: 数据库端口

3. config/config.py:
    前后端使用数据库的配置, 需修改input_jdbc_host值, 如后台服务使用的数据库与算法服务器在同一机器,
    则将其改为127.0.0.1; 否则更改为数据库所在地址

其他:
1. gbase连接方式可能有误, 可参照灵雀云最新版本修改
2. 训练可选择根据大类或小类进行, 预测则在services/main.py 118行中定义通过大类预测,
如需预测小类事件, 请修改此行super_event_col为sub_event_col