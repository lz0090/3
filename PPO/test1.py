import numpy as np
import math as mt
import matplotlib.pyplot as plt
import gym
from AGVTrace import AGV_trace
import torch
import argparse
import config
from normalization import Normalization, RewardScaling
from ppo_model import PPO_discrete


R_ES_to_SO = 15 * np.power(10, 6)        # 根据‘Multitask Multiobjective Deep Reinforcement Learning-Based Computation Offloading Method for Industrial Internet of Things’设置的ES到SO层的传输速率为15Mbits/s
data_ES = 0.1 * np.power(10, 6)          # 设ES传到SO层的数据量为0.1M，因为只是关键数据，所以应该比较小
CPU_cn = 1                               # ‘Multitask Multiobjective Deep Reinforcement Learning-Based Computation Offloading Method for Industrial Internet of Things’中设的每个任务需要的CPU周期为[0.8,8.4]G/task，这里取1G表示SO层计算所需
f_SO_CPU = 30                            # ‘Multitask Multiobjective Deep Reinforcement Learning-Based Computation Offloading Method for Industrial Internet of Things’中设的每个任务需要的CPU周期为[20,40]GHz，这里取30GHz为SO层的计算能力
p_SO_cycle = 0.5                         # 每个CPU周期能耗，这个数值大小还要根据最后的仿真数值来调整

data_trans = 2 * np.power(10, 6)    # 上行传输的数据量为2M
config.glo_data = data_trans
bandwith_ES = 50 * np.power(10, 6)  # 每个ES可分配的带宽为40MHz （可能需要再考虑）
config.glo_band_all = bandwith_ES
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
ES_coordinate_data = np.empty([2, 2], dtype=int)   # 创建边缘节点的二维坐标矩阵
ES_coordinate_data[0][0] = 43  # ES1的x坐标
ES_coordinate_data[0][1] = 44  # ES1的y坐标
ES_coordinate_data[1][0] = 84  # ES2的x坐标
ES_coordinate_data[1][1] = 85  # ES2的y坐标
IPD_coordinate_1 = np.array([(67, 32), (4, 50), (7, 43), (53, 54), (23, 26), (44, 51), (46, 69), (56, 56), (60, 54), (25, 13)])   # ES1内F-IPD的坐标
IPD_coordinate_2 = np.array([(116, 88), (59, 59), (96, 85), (68, 93), (100, 91), (81, 98), (73, 81), (106, 103), (83, 76), (91, 84), (56, 108), (84, 115)]) # ES2内F-IPD的坐标

def Delay_with_M_IPD(data_trans, B, ES_coordinate, num_F, IPD_coordinate, M_coordinate_x):  # 传递的参数分别为上传数据量、所分配的带宽、ES的坐标、所有F-IPD的坐标以及M-IPD的X坐标

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
    for i in range(num_F):
        # gamma[i] = (   (10 ** (P_RX_up[i] / 10)) / (  (10 ** ((N_0 + 10 * mt.log(bandwith_sub * B[i], 10)) / 10)) + (IF - (10 ** (P_RX_up[i] / 10)) + (10 ** (P_RX_up[num_F] / 10)))  )     )
        gamma[i] = (   (10 ** (P_RX_up[i] / 10)) / (  (10 ** ((N_0 + 10 * mt.log(bandwith_sub * B[i], 10)) / 10))  )     )           #     F-IPDs的信噪比
    # gamma[num_F] = (   (10 ** (P_RX_up[num_F] / 10)) / (  (10 ** ((N_0 + 10 * mt.log(bandwith_sub * B[num_F], 10)) / 10)) + IF  )     )
    gamma[num_F] = (   (10 ** (P_RX_up[num_F] / 10)) / (  (10 ** ((N_0 + 10 * mt.log(bandwith_sub * B[num_F], 10)) / 10)) )     )    #     M-IPD的信噪比

    for i in range(num_F):
        T_up[i] =  data_trans / (bandwith_sub * B[i] * mt.log(1 + gamma[i], 2))  # 求出ES1内的F-IPD的上行链路时延
    T_up[num_F] = (0.8 * data_trans) / (bandwith_sub * B[num_F] * mt.log(1 + gamma[num_F], 2)) # 求出ES1内的M-IPD的上行链路时延


    return T_up

def Delay(data_trans, B, ES_coordinate, num_F, IPD_coordinate):  # 传递的参数分别为上传数据量、所分配的带宽、ES的坐标、所有F-IPD的坐标以及M-IPD的X坐标

    gamma_dB = np.zeros([(num_F), 1], dtype=np.float32)     # 创建上行链路ES处的信噪比γ一维矩阵(dB形式)，最后一个表示M-IPD
    gamma = np.zeros([(num_F), 1], dtype=np.float32)        # 创建上行链路ES处的信噪比γ一维矩阵，最后一个表示M-IPD
    P_RX_up = np.zeros([(num_F), 1], dtype=np.float32)      # 创建上行链路ES处的接收功率一维矩阵(dB形式)
    T_up = np.zeros([(num_F), 1], dtype=np.float32)         # 创建上行链路时延一维矩阵
    IF = 0                                                      # IF表示所有F-IPD的到达ES1处的功率

    "求上行链路时延"
    for i in range(num_F):
        P_RX_up[i] = power_tx_F - (20 * mt.log(((4 * mt.pi * f_0 * d_0) / c_0), 10) + \
                     10 * n_loss_F * mt.log(mt.sqrt(mt.pow(IPD_coordinate[i][0] - ES_coordinate[0], 2) + mt.pow(IPD_coordinate[i][1] - ES_coordinate[1], 2)) / d_0, 10) + xigema_shadow)  # 求ES1范围内F-IPD的dB形式的上行链路接收功率

    for i in range(num_F):
        # gamma[i] = (   (10 ** (P_RX_up[i] / 10)) / (  (10 ** ((N_0 + 10 * mt.log(bandwith_sub * B[i], 10)) / 10)) + (IF - (10 ** (P_RX_up[i] / 10)) + (10 ** (P_RX_up[num_F] / 10)))  )     )
        gamma[i] = (   (10 ** (P_RX_up[i] / 10)) / (  (10 ** ((N_0 + 10 * mt.log(bandwith_sub * B[i], 10)) / 10))  )     )           #     F-IPDs的信噪比

    for i in range(num_F):
        T_up[i] =  data_trans / (bandwith_sub * B[i] * mt.log(1 + gamma[i], 2))  # 求出ES1内的F-IPD的上行链路时延
    return T_up


np_arr_ES1 = np.load('./np_arr_ES1.npy', allow_pickle=True)
np_arr_ES2 = np.load('./np_arr_ES2.npy', allow_pickle=True)
A = 120                                               # 将时延单位标准化为标准成本单位的值
B = 90                                                 # 将能耗单位标准化为标准成本单位的值
w_1 = 0.5                                              # 时延成本对应的权重
w_2 = 0.9                                              # 能耗成本对应的权重
M_coordinate_x = 23.0
v_M = 2
T = 100
T_run_1 = 52  # M-IPD在ES1内的运动时隙数
E_trans_ES1 = 0  # 定义ES1传输能耗
E_trans_ES2 = 0  # 定义ES2传输能耗
E_trans = 0  # 定义ES1传输能耗
'在所有时隙都进行资源的重新分配'
Delay_ES = np.zeros([T, 1], dtype=np.float32)                                         # 定义所有时隙的总时延
Energy_ES = np.zeros([T, 1], dtype=np.float32)                                        # 定义所有时隙的总能耗（不包括重新分配资源时的计算能耗）
cost = np.zeros([T, 1], dtype=np.float32)                                               # 定义所有时隙的时延能耗成本（不包括重新分配资源时的计算能耗）
for t in range(T_run_1):
    E_trans_ES1 = 0  # 定义ES1传输能耗
    E_trans_ES2 = 0  # 定义ES2传输能耗
    M_x = M_coordinate_x + v_M * t * (50 / T)  # 当前时隙M-IPD的位置
    T_trans_ES1 = Delay_with_M_IPD(data_trans, np_arr_ES1[t], ES_coordinate_data[0], 10, IPD_coordinate_1, M_x)
    T_trans_ES2 = Delay(data_trans, np_arr_ES2[t], ES_coordinate_data[1], 12, IPD_coordinate_2)
    Delay_ES[t] = sum(T_trans_ES1) + sum(T_trans_ES2)
    for i in range(10):
        E_trans_ES1 += ((10 ** (power_tx_F / 10)) / 1000) * T_trans_ES1[i]  # F-IPDs的传输能耗和
    E_trans_ES1 += ((10 ** (power_tx_M / 10)) / 1000) * T_trans_ES1[10] + ((10 ** (power_tx_ES / 10)) / 1000) * ( data_ES / R_ES_to_SO)
    for i in range(12):
        E_trans_ES2 += ((10 ** (power_tx_F / 10)) / 1000) * T_trans_ES2[i]  # F-IPDs的传输能耗和
    E_trans_ES2 +=((10 ** (power_tx_ES / 10)) / 1000) * ( data_ES / R_ES_to_SO)
    Energy_ES[t] = E_trans_ES1 + E_trans_ES2 + p_SO_cycle * CPU_cn * 2 + 1


for t in range(T_run_1, T):
    E_trans_ES1 = 0  # 定义ES1传输能耗
    E_trans_ES2 = 0  # 定义ES2传输能耗
    M_x = M_coordinate_x + v_M * t * (50 / T)  # 当前时隙M-IPD的位置
    T_trans_ES2 = Delay_with_M_IPD(data_trans, np_arr_ES2[t], ES_coordinate_data[1], 12, IPD_coordinate_2, M_x)
    T_trans_ES1 = Delay(data_trans, np_arr_ES1[t], ES_coordinate_data[0], 10, IPD_coordinate_1)
    Delay_ES[t] = sum(T_trans_ES1) + sum(T_trans_ES2)
    for i in range(12):
        E_trans_ES2 += ((10 ** (power_tx_F / 10)) / 1000) * T_trans_ES2[i]  # F-IPDs的传输能耗和
    E_trans_ES2 += ((10 ** (power_tx_M / 10)) / 1000) * T_trans_ES2[12] + ((10 ** (power_tx_ES / 10)) / 1000) * ( data_ES / R_ES_to_SO)
    for i in range(10):
        E_trans_ES1 += ((10 ** (power_tx_F / 10)) / 1000) * T_trans_ES1[i]  # F-IPDs的传输能耗和
    E_trans_ES1 +=((10 ** (power_tx_ES / 10)) / 1000) * ( data_ES / R_ES_to_SO)
    Energy_ES[t] = E_trans_ES1 + E_trans_ES2 + p_SO_cycle * CPU_cn * 2 + 1
cost = w_1 * A * Delay_ES + (1 - w_1) * B * Energy_ES
print(sum(cost) / T)
print((w_1 * 1 * sum(Delay_ES)) / T)
print(((1 - w_1) * 1 * sum(Energy_ES)) / T)
