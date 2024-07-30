import random
import numpy as np
import math as mt
import matplotlib.pyplot as plt

# 生成给定数目的随机二维坐标，coverage_server是边缘服务器的覆盖范围，num_PM是设备节点数目
def generate_random_coordinates(coverage_server, num_PM):
    PM_coordinate = np.array([[0, 0]])   # 预置一个坐标确定PM_coordinate的形状
    while(len(PM_coordinate) < (num_PM + 1)):

        x = random.randint(coverage_server, 100 + coverage_server)
        y = random.randint(coverage_server, 100 + coverage_server)
        tmp = np.array([[x, y]])
        if tmp not in PM_coordinate:  # 判断是否存在重复坐标
            PM_coordinate = np.vstack((PM_coordinate, tmp))

    PM_coordinate = np.delete(PM_coordinate, 0, axis=0)  # 删除开始预置的坐标

    return PM_coordinate

# coverage_server = 30
# num_PM = 120
# PM_coordinate = generate_random_coordinates(coverage_server, num_PM)
# print(PM_coordinate)
# print(len(PM_coordinate))
# for i in range(num_PM):
#     plt.plot(PM_coordinate[i][0], PM_coordinate[i][1], 'co')   # 对边缘节点1内的IIE的坐标位置加粗显示
# plt.show()             # 对设计的场景进行绘图