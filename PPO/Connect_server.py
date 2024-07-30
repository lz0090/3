import numpy as np
import math as mt
import random
import matplotlib.pyplot as plt

# PM_coordinate是设备节点坐标，num_PM是设备节点数目，num_server是边缘服务器数目，coverage_server是边缘服务器覆盖范围，action是动作值即边缘服务器坐标
def connect_server_judgment(PM_coordinate, num_PM, num_server, coverage_server, action):
    PM_allocation = []        # 创建节点分配数组，里面的值表示设备节点连接的边缘服务器如0,1,2...
    flag_connect = 0          # 代表设备节点连通性的变量，有一个设备节点具备了连通性就+1
    for j in range(num_PM):
        PM_distance = []      # 创建设备节点j和每个边缘服务器之间距离数组，用于根据距离最小和连通性来判断应该连接的边缘节点
        for i in range(num_server):
            tmp1 = (mt.pow(action[2 * i] - PM_coordinate[j][0], 2) + mt.pow(action[2 * i + 1] - PM_coordinate[j][1], 2))    # 中间值存储设备节点j和边缘服务器之间的距离
            PM_distance.append(tmp1)

        if min(PM_distance) <= coverage_server ** 2:   # 判断设备节点i和距离最近的边缘服务器之间是否具备连通性
            flag_connect += 1
            tmp2 = PM_distance.index(min(PM_distance))  # tmp2即对应的距离最小的边缘服务器
            PM_allocation.append(tmp2)
            PM_distance = []

    return flag_connect, PM_allocation         # 返回flag_connect的目的是为了根据flag_connect的值设置递进式奖励






















