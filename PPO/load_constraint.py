import numpy as np
import math as mt
import random
import matplotlib.pyplot as plt


# num_server是边缘节点数，PM_allocation是节点分配数组，里面的值表示设备节点连接的边缘服务器如0,1,2... , data_comp是每个生产设备的计算任务量，C_max是单个IES最大计算能力，load_eta是负载均衡约束阈值
def load_judgment(num_server, PM_allocation, data_comp, C_max, load_eta):
    server_load = []  # 创建边缘服务器负载列表，用于存储每个边缘服务器的负载值
    ######  计算负载
    for i in range(num_server):
        index = [k for k, val in enumerate(PM_allocation) if val == i]  # 找出PM_allocation中哪些位置值为i（i即为对应连接的边缘服务器）
        tmp_load = 0  # 中间存储变量
        for j in range(len(index)):
            tmp_load += data_comp  # 计算边缘服务器i的负载
        server_load.append(tmp_load)  # 将边缘服务器i的负载数据添加到列表中
    return server_load

