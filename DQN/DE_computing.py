import random
import numpy as np
import math as mt

import config
from AGVTrace import AGV_trace


bandwith_ES = config.glo_band_all  # 每个ES可分配的带宽为40MHz （可能需要再考虑）
num_sub_max = 50  # 每个边缘节点能够划分的最大子载波数为50
bandwith_sub = bandwith_ES/num_sub_max # 每个子载波的带宽
f_0 = 2.4 * np.power(10, 9)   # 理应是每个子载波中心频率都不一样，但中心频率远大于可分配的带宽，因此差异可以忽略不计，此外不同边缘节点中心频率应该是不一样，但考虑到差异较小，可以进行忽略
N_0 = -174      # 高斯白噪声功率谱密度为-174dBm/Hz  4 * np.float_power(10, -21)
d_0 = 1  # 路径损耗参考距离通常取1m
c_0 = 3 * np.power(10, 8) # 光速
n_loss_F = 1.69 # F-IPD路径损耗指数，参考ll论文生产装配车间
n_loss_M = 4  # M-IPD路径损耗指数
xigema_shadow = 3.39  # 阴影衰落3.39dB，参考ll论文生产装配车间
power_tx_ES = 35      # ES发射功率一样，均为35dBm
power_tx_F = 10       # F-IPD发射功率均为10dBm
power_tx_M = 17       # M-IPD发射功率均为17dBm
IPD_coordinate_all = np.array([(67, 32), (4, 50), (7, 43), (53, 54), (23, 26), (44, 51), (46, 69), (56, 56), (60, 54), (25, 13), (116, 88), (59, 59), (96, 85), (68, 93), (100, 91), (81, 98), (73, 81), (106, 103), (83, 76), (91, 84), (56, 108), (84, 115)])



def Delay(data_trans, B, ES_coordinate, num_F, IPD_coordinate, M_coordinate_x):  # 传递的参数分别为上传数据量、所分配的带宽、ES的坐标、所有F-IPD的坐标以及M-IPD的X坐标

    gamma_dB = np.zeros([(num_F + 1), 1], dtype=np.float32)     # 创建上行链路ES处的信噪比γ一维矩阵(dB形式)，最后一个表示M-IPD
    gamma = np.zeros([(num_F + 1), 1], dtype=np.float32)        # 创建上行链路ES处的信噪比γ一维矩阵，最后一个表示M-IPD
    P_RX_up = np.zeros([(num_F + 1), 1], dtype=np.float32)      # 创建上行链路ES处的接收功率一维矩阵(dB形式)
    T_up = np.zeros([(num_F + 1), 1], dtype=np.float32)         # 创建上行链路时延一维矩阵
    IF = 0                                                      # IF表示所有F-IPD的到达ES1处的功率

    "求上行链路时延"
    for i in range(num_F):
        P_RX_up[i] = power_tx_F - (20 * mt.log(((4 * mt.pi * f_0 * d_0) / c_0), 10) + \
                     10 * n_loss_F * mt.log(mt.sqrt(mt.pow(IPD_coordinate[i][0] - ES_coordinate[0], 2) + mt.pow(IPD_coordinate[i][1] - ES_coordinate[1], 2)) / d_0, 10) + xigema_shadow)  # 求ES1范围内F-IPD的dB形式的上行链路接收功率
    P_RX_up[num_F] = power_tx_M - (20 * mt.log(((4 * mt.pi * f_0 * d_0) / c_0), 10) + \
                    10 * n_loss_M * mt.log(mt.sqrt(mt.pow(abs(M_coordinate_x - ES_coordinate[0]), 2) + mt.pow(abs(AGV_trace(M_coordinate_x) - ES_coordinate[1]), 2)) / d_0, 10) + xigema_shadow)      # 求ES1范围内M-IPD的dB形式的上行链路接收功率

    # for i in range(len(IPD_coordinate_all)):    # 所有F-IPDs包括ES1和ES2内的信号到ES1处的功率汇总
    #     IF += 10 ** ((power_tx_F - (20 * mt.log(((4 * mt.pi * f_0 * d_0) / c_0), 10) + \
    #                  10 * n_loss_F * mt.log(mt.sqrt(mt.pow(IPD_coordinate_all[i][0] - ES_coordinate[0], 2) + mt.pow(IPD_coordinate_all[i][1] - ES_coordinate[1], 2)) / d_0, 10) + xigema_shadow)) / 10)  # 求ES1范围内F-IPD的dB形式的上行链路接收功率

    for i in range(num_F):
        # gamma[i] = (   (10 ** (P_RX_up[i] / 10)) / (  (10 ** ((N_0 + 10 * mt.log(bandwith_sub * B[i], 10)) / 10)) + (IF - (10 ** (P_RX_up[i] / 10)) + (10 ** (P_RX_up[num_F] / 10)))  )     )
        gamma[i] = (   (10 ** (P_RX_up[i] / 10)) / (  (10 ** ((N_0 + 10 * mt.log(bandwith_sub * B[i], 10)) / 10))  )     )           #     F-IPDs的信噪比
    # gamma[num_F] = (   (10 ** (P_RX_up[num_F] / 10)) / (  (10 ** ((N_0 + 10 * mt.log(bandwith_sub * B[num_F], 10)) / 10)) + IF  )     )
    gamma[num_F] = (   (10 ** (P_RX_up[num_F] / 10)) / (  (10 ** ((N_0 + 10 * mt.log(bandwith_sub * B[num_F], 10)) / 10)) )     )    #     M-IPD的信噪比

    for i in range(num_F):
        T_up[i] =  data_trans / (bandwith_sub * B[i] * mt.log(1 + gamma[i], 2))  # 求出ES1内的F-IPD的上行链路时延
    T_up[num_F] = (0.8 * data_trans) / (bandwith_sub * B[num_F] * mt.log(1 + gamma[num_F], 2)) # 求出ES1内的M-IPD的上行链路时延


    return T_up

# ES_coordinate_data = np.empty([2, 2], dtype=int)   # 创建边缘节点的二维坐标矩阵
# ES_coordinate_data[0][0] = 43  # ES1的x坐标
# ES_coordinate_data[0][1] = 44  # ES1的y坐标
# ES_coordinate_data[1][0] = 84  # ES2的x坐标
# ES_coordinate_data[1][1] = 85  # ES2的y坐标
# ES_coordinate = ES_coordinate_data[1]
# IPD_coordinate = np.array([(116, 88), (59, 59), (96, 85), (68, 93), (100, 91), (81, 98), (73, 81), (106, 103), (83, 76), (91, 84), (56, 108), (84, 115)]) # ES2内F-IPD的坐标
# a = Delay(2 * np.power(10, 6), [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 24], ES_coordinate, 12, IPD_coordinate, 121)
# print(a)