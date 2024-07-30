
# coding:utf-8
import math
import numpy as np
import math as mt
import random
import matplotlib.pyplot as plt
import gym
from gym import spaces
from AGVTrace import AGV_trace
from DE_computing_FIPDs import Delay
import config_MIPD


'M-IPD在ES1内时ES2的环境'
' 系统环境变量 '
# ****************************************************************************************************

num_ES = 2  # 边缘节点数
num_IPD_array_1 = 10  # 边缘节点1内的F-IPD数目
num_IPD_array_2 = 12  # 边缘节点2内的F-IPD数目
num_IPD = num_IPD_array_1 + num_IPD_array_2    # F-IPD数目的总和（不包括M-IPD）
coverage_ES = 40  # 每个边缘节点的覆盖范围为40m
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

bandwith_ES = config_MIPD.glo_band_all  # 每个ES可分配的带宽为40MHz （可能需要再考虑）
num_slot = 50    # 时隙数 （需要再考虑）
num_sub_max = 50  # 每个边缘节点能够划分的最大子载波数为50
bandwith_sub = bandwith_ES/num_sub_max # 每个子载波的带宽
T_F_max = 0.10 # 对于不动的F-IPD，每条链路能够允许的最大上下行时延和 0.10
T_M_max = 0.02 # 对于M-IPD，其通信链路能够允许的最大上下行时延和 0.02
v_M = 2           # M-IPD速度为2m/s (为x轴方向的分速度为2m/s)
data_trans = config_MIPD.glo_data  # 上行传输的数据量为2M

ES_coordinate_data = np.empty([num_ES, 2], dtype=int)   # 创建边缘节点的二维坐标矩阵
ES_coordinate_data[0][0] = 43  # ES1的x坐标
ES_coordinate_data[0][1] = 44  # ES1的y坐标
ES_coordinate_data[1][0] = 84  # ES2的x坐标
ES_coordinate_data[1][1] = 85  # ES2的y坐标
IPD_coordinate_1 = np.array([(67, 32), (4, 50), (7, 43), (53, 54), (23, 26), (44, 51), (46, 69), (56, 56), (60, 54), (25, 13)])   # ES1内F-IPD的坐标
IPD_coordinate_2 = np.array([(116, 88), (59, 59), (96, 85), (68, 93), (100, 91), (81, 98), (73, 81), (106, 103), (83, 76), (91, 84), (56, 108), (84, 115)]) # ES2内F-IPD的坐标
IPD_coordinate_all = np.array([(67, 32), (4, 50), (7, 43), (53, 54), (23, 26), (44, 51), (46, 69), (56, 56), (60, 54), (25, 13), (116, 88), (59, 59), (96, 85), (68, 93), (100, 91), (81, 98), (73, 81), (106, 103), (83, 76), (91, 84), (56, 108), (84, 115)])  # 所有F-IPD坐标的集合


###################   可调参数
num_F = num_IPD_array_2                  # 选择当前仿真的F-IPD数目，num_IPD_array_1代表ES1内的F-IPD数目，num_IPD_array_2代表ES2内的
ES_coordinate = ES_coordinate_data[1]    # 当前选择的ES，0表示ES1，1表示ES2
IPD_coordinate = IPD_coordinate_2        # 当前选择的ES内的F-IPD的坐标
EXTRA_num_sub = num_sub_max - (num_F)    # 因为每个设备都至少需要分配一个子载波，因此先给每个设备都预分配1个子载波，然后算法分配剩余子载波
R_ES_to_SO = 15 * np.power(10, 6)        # 根据‘Multitask Multiobjective Deep Reinforcement Learning-Based Computation Offloading Method for Industrial Internet of Things’设置的ES到SO层的传输速率为15Mbits/s
data_ES = 0.1 * np.power(10, 6)          # 设ES传到SO层的数据量为0.1M，因为只是关键数据，所以应该比较小
CPU_cn = 1                               # ‘Multitask Multiobjective Deep Reinforcement Learning-Based Computation Offloading Method for Industrial Internet of Things’中设的每个任务需要的CPU周期为[0.8,8.4]G/task，这里取1G表示SO层计算所需
f_SO_CPU = 30                            # ‘Multitask Multiobjective Deep Reinforcement Learning-Based Computation Offloading Method for Industrial Internet of Things’中设的每个任务需要的CPU周期为[20,40]GHz，这里取30GHz为SO层的计算能力
p_SO_cycle = 0.5                         # 每个CPU周期能耗，这个数值大小还要根据最后的仿真数值来调整

lambda_error = 0.4                       # 预测误差对重分配决策的影响参数
#  ****************************************************************************************************



class MyEnv1(gym.Env):
    def __init__(self):
        self.viewer = None                # 初始化赋值
        self.num_ES = num_ES
        self.ES_coordinate = ES_coordinate
        self.IPD_coordinate = IPD_coordinate
        self.bandwith_ES = bandwith_ES
        self.coverage_ES = coverage_ES
        self.N_0 = N_0
        self.num_slot = num_slot
        self.power_tx_ES = power_tx_ES
        self.power_tx_F = power_tx_F
        self.power_tx_M = power_tx_M
        self.num_sub_max = num_sub_max
        self.v_M = v_M
        self.allocation = [2] * num_F                    # 初始分配，每个设备都至少分配一个子载波

        self.action_space = spaces.Discrete(num_F)       # 离散取值动作空间，动作空间的大小是ES内的设备数目（F-IPD+M-IPD）当前选择ES1
        self.observation_space = gym.spaces.Box(low=1, high=num_sub_max, shape=(num_F,), dtype=np.int32)  # 状态空间取值为每个设备可以分配的载波数，最小为1，最大为最大子载波数50

    def step(self, action):
        reward = 0  # 定义初始奖励值为0
        T_trans_old = 0
        T_all_old = 0
        E_trans_old = 0
        E_all_old = 0
        T_trans_new = 0
        T_all_new = 0
        E_trans_new = 0
        E_all_new = 0
        flag_M = 0  # 判断M-IPD链路时延是否满足要求，满足则取1，不满足取0
        flag_F = 0  # 判断M-IPD链路时延是否满足要求，每有一个设备满足则加1
        reward_temporary = 0 # 定义一个中间过程的奖励值，用于最后的奖励求和
        done = False  # 判断一个episode是否结束

        # 算当前状态下时延和能耗
        T_trans_old = Delay(data_trans, self.allocation, ES_coordinate, num_F, IPD_coordinate)                             # 用delay函数计算出每个设备此时的数据传输时延
        T_all_old = sum(T_trans_old) + data_ES / R_ES_to_SO * num_F + CPU_cn / f_SO_CPU                                        # 总时延为IPDs传输时延加上ES到SO层传输时延和SO层计算时延
        for i in range(num_F):
            E_trans_old += ((10 ** (power_tx_F / 10)) / 1000) * T_trans_old[i]                                                                      # F-IPDs的传输能耗和
        E_trans_old += ((10 ** (power_tx_ES / 10)) / 1000) * (data_ES / R_ES_to_SO)                                # 加上M-IPD的传输能耗和ES传输到SO的能耗
        E_all_old = E_trans_old + p_SO_cycle * CPU_cn                                                                        # 加上SO层的计算能耗   '还没有加上重新分配的能耗'

        ############################ 奖励函数部分

        self.allocation[action] += 1                                                                                   # 当M-IPD时延满足要求时只有动作值不取到num_F时才能执行分配动作，为F-IPD分配一个子载波
        next_state = self.allocation                                                                                   # 下一个状态为执行动作后的设备带宽分配情况
        T_trans_new = Delay(data_trans, self.allocation, ES_coordinate, num_F, IPD_coordinate)                          # 用delay函数计算出每个设备此时的数据传输时延，数组最后一个表示M-IPD
        T_all_new = sum(T_trans_new) + data_ES / R_ES_to_SO + CPU_cn / f_SO_CPU                                                # 总时延为IPDs传输时延加上ES到SO层传输时延和SO层计算时延
        for i in range(num_F):
            E_trans_new += ((10 ** (power_tx_F / 10)) / 1000) * T_trans_new[i]                                                 # F-IPDs的传输能耗和
        E_trans_new += ((10 ** (power_tx_ES / 10)) / 1000) * (data_ES / R_ES_to_SO)                                        # 加上M-IPD的传输能耗和ES传输到SO的能耗
        E_all_new = E_trans_new + p_SO_cycle * CPU_cn                                                                          # 加上SO层的计算能耗   '还没有加上重新分配的能耗'
        reward = (T_all_old - T_all_new) + (E_all_old - E_all_new)
        if sum( self.allocation) == num_sub_max:
            done = True
        else:
            reward -= -100


        info = {'T': [sum(T_trans_new), E_all_new,self.allocation, sum(self.allocation)]}



        return next_state, reward, done, info

    def reset(self):
        self.allocation = [2] * num_F                                                                                # 初始分配，每个设备都至少分配一个子载波
        return np.array(self.allocation)


    def render(self, mode="human"):
        return None

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None



# 'M-IPD在ES2内，ES2的myenv.py'
# ' 系统环境变量 '
# # ****************************************************************************************************
# 
# num_ES = 2  # 边缘节点数
# num_IPD_array_1 = 10  # 边缘节点1内的F-IPD数目
# num_IPD_array_2 = 12  # 边缘节点2内的F-IPD数目
# num_IPD = num_IPD_array_1 + num_IPD_array_2    # F-IPD数目的总和（不包括M-IPD）
# coverage_ES = 40  # 每个边缘节点的覆盖范围为40m
# f_0 = 2.4 * np.power(10, 9)   # 理应是每个子载波中心频率都不一样，但中心频率远大于可分配的带宽，因此差异可以忽略不计，此外不同边缘节点中心频率应该是不一样，但考虑到差异较小，可以进行忽略
# N_0 = -174      # 高斯白噪声功率谱密度为-174dBm/Hz  4 * np.float_power(10, -21)
# d_0 = 1  # 路径损耗参考距离通常取1m
# c_0 = 3 * np.power(10, 8) # 光速
# n_loss_F = 1.69 # F-IPD路径损耗指数，参考ll论文生产装配车间
# n_loss_M = 4  # M-IPD路径损耗指数
# xigema_shadow = 3.39  # 阴影衰落3.39dB，参考ll论文生产装配车间
# power_tx_ES = 35      # ES发射功率一样，均为35dBm
# power_tx_F = 10       # F-IPD发射功率均为10dBm
# power_tx_M = 17       # M-IPD发射功率均为17dBm
# 
# bandwith_ES = 40 * np.power(10, 6)  # 每个ES可分配的带宽为40MHz （可能需要再考虑）
# num_slot = 50    # 时隙数 （需要再考虑）
# num_sub_max = 50  # 每个边缘节点能够划分的最大子载波数为50
# bandwith_sub = bandwith_ES/num_sub_max # 每个子载波的带宽
# T_F_max = 0.10 # 对于不动的F-IPD，每条链路能够允许的最大上下行时延和 0.10
# T_M_max = 0.02 # 对于M-IPD，其通信链路能够允许的最大上下行时延和 0.02
# v_M = 2           # M-IPD速度为2m/s (为x轴方向的分速度为2m/s)
# data_trans = 2 * np.power(10, 6)  # 上行传输的数据量为2M
# 
# ES_coordinate_data = np.empty([num_ES, 2], dtype=int)   # 创建边缘节点的二维坐标矩阵
# ES_coordinate_data[0][0] = 43  # ES1的x坐标
# ES_coordinate_data[0][1] = 44  # ES1的y坐标
# ES_coordinate_data[1][0] = 84  # ES2的x坐标
# ES_coordinate_data[1][1] = 85  # ES2的y坐标
# IPD_coordinate_1 = np.array([(67, 32), (4, 50), (7, 43), (53, 54), (23, 26), (44, 51), (46, 69), (56, 56), (60, 54), (25, 13)])   # ES1内F-IPD的坐标
# IPD_coordinate_2 = np.array([(116, 88), (59, 59), (96, 85), (68, 93), (100, 91), (81, 98), (73, 81), (106, 103), (83, 76), (91, 84), (56, 108), (84, 115)]) # ES2内F-IPD的坐标
# IPD_coordinate_all = np.array([(67, 32), (4, 50), (7, 43), (53, 54), (23, 26), (44, 51), (46, 69), (56, 56), (60, 54), (25, 13), (116, 88), (59, 59), (96, 85), (68, 93), (100, 91), (81, 98), (73, 81), (106, 103), (83, 76), (91, 84), (56, 108), (84, 115)])  # 所有F-IPD坐标的集合
# 
# 
# ###################   可调参数
# num_F = num_IPD_array_2                 # 选择当前仿真的F-IPD数目，num_IPD_array_1代表ES1内的F-IPD数目，num_IPD_array_2代表ES2内的
# ES_coordinate = ES_coordinate_data[1]    # 当前选择的ES，0表示ES1，1表示ES2
# IPD_coordinate = IPD_coordinate_2        # 当前选择的ES内的F-IPD的坐标
# PRIORITY_IPD = num_IPD_array_2           # 定义M-IPD的编号为动作空间中的最后一个取值在这里即为12，要优先给M-IPD进行带宽资源的分配
# EXTRA_num_sub = num_sub_max - (num_IPD_array_2 + 1)  # 因为每个设备都至少需要分配一个子载波，因此先给每个设备都预分配1个子载波，然后算法分配剩余子载波
# R_ES_to_SO = 15 * np.power(10, 6)        # 根据‘Multitask Multiobjective Deep Reinforcement Learning-Based Computation Offloading Method for Industrial Internet of Things’设置的ES到SO层的传输速率为15Mbits/s
# data_ES = 0.1 * np.power(10, 6)          # 设ES传到SO层的数据量为0.1M，因为只是关键数据，所以应该比较小
# CPU_cn = 1                               # ‘Multitask Multiobjective Deep Reinforcement Learning-Based Computation Offloading Method for Industrial Internet of Things’中设的每个任务需要的CPU周期为[0.8,8.4]G/task，这里取1G表示SO层计算所需
# f_SO_CPU = 30                            # ‘Multitask Multiobjective Deep Reinforcement Learning-Based Computation Offloading Method for Industrial Internet of Things’中设的每个任务需要的CPU周期为[20,40]GHz，这里取30GHz为SO层的计算能力
# p_SO_cycle = 2                           # 每个CPU周期能耗，这个数值大小还要根据最后的仿真数值来调整
# 
# lambda_error = 0.4                       # 预测误差对重分配决策的影响参数
# #  ****************************************************************************************************
# 
# 
# 
# class MyEnv(gym.Env):
#     def __init__(self):
#         self.viewer = None                # 初始化赋值
#         self.num_ES = num_ES
#         self.ES_coordinate = ES_coordinate
#         self.IPD_coordinate = IPD_coordinate
#         self.bandwith_ES = bandwith_ES
#         self.coverage_ES = coverage_ES
#         self.N_0 = N_0
#         self.num_slot = num_slot
#         self.power_tx_ES = power_tx_ES
#         self.power_tx_F = power_tx_F
#         self.power_tx_M = power_tx_M
#         self.num_sub_max = num_sub_max
#         self.v_M = v_M
#         self.allocation = [1] * (num_F + 1)  # 初始分配，每个设备都至少分配一个子载波
# 
#         self.action_space = spaces.Discrete(num_F + 1)   # 离散取值动作空间，动作空间的大小是ES内的设备数目（F-IPD+M-IPD）当前选择ES1
#         self.observation_space = gym.spaces.Box(low=1, high=num_sub_max, shape=((num_F + 1),), dtype=np.int32)  # 状态空间取值为每个设备可以分配的载波数，最小为1，最大为最大子载波数50
# 
#     def step(self, action):
# 
#         M_coordinate_x = config.glo_M_x  # M-IPD初始x坐标为23,初始y坐标可由AGV的运动轨迹y=x^0.9+1/10*x算出，通过config文件导入变量值
#         reward = 0  # 定义初始奖励值为0
#         T_all = 0   # 定义总时延变量
#         E_all = 0   # 定义总能耗变量
#         E_trans = 0 # 定义传输能耗
#         flag_M = 0  # 判断M-IPD链路时延是否满足要求，满足则取1，不满足取0
#         flag_F = 0  # 判断M-IPD链路时延是否满足要求，每有一个设备满足则加1
#         reward_temporary = 0 # 定义一个中间过程的奖励值，用于最后的奖励求和
#         done = False  # 判断一个episode是否结束
# 
#         # 算当前状态下时延和能耗
#         T_trans = Delay(data_trans, self.allocation, ES_coordinate, num_F, IPD_coordinate, M_coordinate_x)           # 用delay函数计算出每个设备此时的数据传输时延，数组最后一个表示M-IPD
#         T_all = sum(T_trans) + data_ES / R_ES_to_SO * num_F + CPU_cn / f_SO_CPU                                      # 总时延为IPDs传输时延加上ES到SO层传输时延和SO层计算时延
#         # for i in range(num_F):
#         #     E_trans += power_tx_F * T_trans[i]                                                                       # F-IPDs的传输能耗和
#         # E_trans += power_tx_M * T_trans[num_F] + power_tx_ES * (data_ES / R_ES_to_SO)                                # 加上M-IPD的传输能耗和ES传输到SO的能耗
#         # E_all = E_trans + p_SO_cycle * CPU_cn                                                                        # 加上SO层的计算能耗   '还没有加上重新分配的能耗'
# 
#         ############################ 奖励函数部分
# 
# 
#         if T_trans[num_F] <= T_M_max and action != num_F:                                                            # M-IPD链路的时延满足要求,对每一个F-IPD的时延进行判断，奖励与满足时延约束的设备数成正比
#             flag_M = 1                                                                                               # M-IPD时延满足要求，flag_M置1
# 
#             self.allocation[action] += 1                                                                             # 当M-IPD时延满足要求时只有动作值不取到num_F时才能执行分配动作，为F-IPD分配一个子载波
#             next_state = self.allocation                                                                             # 下一个状态为执行动作后的设备带宽分配情况
# 
#             T_trans = Delay(data_trans, self.allocation, ES_coordinate, num_F, IPD_coordinate, M_coordinate_x)       # 用delay函数计算出每个设备此时的数据传输时延，数组最后一个表示M-IPD
#             T_all = sum(T_trans) + data_ES / R_ES_to_SO + CPU_cn / f_SO_CPU                                          # 总时延为IPDs传输时延加上ES到SO层传输时延和SO层计算时延
#             for i in range(num_F):
#                 E_trans += ((10 ** (power_tx_F / 10)) / 1000) * T_trans[i]                                                                   # F-IPDs的传输能耗和
#             E_trans += ((10 ** (power_tx_M / 10)) / 1000) * T_trans[num_F] + ((10 ** (power_tx_ES / 10)) / 1000) * (data_ES / R_ES_to_SO)                            # 加上M-IPD的传输能耗和ES传输到SO的能耗
#             E_all = E_trans + p_SO_cycle * CPU_cn                                                                    # 加上SO层的计算能耗   '还没有加上重新分配的能耗'
# 
#             for i in range(num_F):
#                 if T_trans[i] <= T_F_max:
#                     flag_F += 1                                                                                      # 每有一个F-IPD设备时延满足要求，flag_F加1
#                     reward += (T_F_max - T_trans[i]) * 20
#                     # reward += 1                                                                                    # 每有一个F-IPD设备时延满足要求，reward加1
#                 else:
#                     reward += (T_F_max - T_trans[i]) * 50
#                     # reward -= 2                                                                                    # 每有一个F-IPD时延不满足要求，reward减少
#         elif T_trans[num_F] <= T_M_max and action == num_F:                                                          # 当M-IPD时延满足要求时动作值取到num_F时不执行分配动作
#             flag_M = 1                                                                                               # M-IPD时延满足要求，flag_M置1
#             next_state = self.allocation                                                                             # 下一个状态为执行动作后的设备带宽分配情况
#             reward += -100                                                                                              # M-IPD满足时延约束了动作值取到num_F会获得惩罚，且不分配带宽
# 
# 
# 
#         elif action == num_F and T_trans[num_F] > T_M_max:
#             self.allocation[action] += 1                                                                            # 当M-IPD时延不满足要求时只有动作值取到num_F时才能执行分配动作，为M-IPD分配一个子载波
#             next_state = self.allocation                                                                            # 下一个状态为执行动作后的设备带宽分配情况
#             reward += 50                                                                                            # 若M-IPD时延不满足要求，则给M-IPD分配带宽会获得较大奖励
# 
#         else:
#             next_state = self.allocation  # 下一个状态为执行动作后的设备带宽分配情况
#             reward = -5 if T_trans[num_F] > T_M_max else 1                                                          # 动作值取到为非优先的F-IPD会获得惩罚，且不分配带宽
# 
#         if  flag_M == 1 and flag_F == num_F:
#             reward += ((num_F * T_F_max + T_M_max) / sum(T_trans)) * 400
#             reward -= abs((sum(self.allocation) - num_sub_max)) * 200
#             done = True
# 
#         info = {'T': [T_trans, self.allocation, sum(T_trans)]}
# 
# 
# 
#         return next_state, reward, done, info
# 
#     def reset(self):
#         self.allocation = [1] * (num_F + 1)                                                                        # 初始分配，每个设备都至少分配一个子载波
#         return np.array(self.allocation)
# 
# 
#     def render(self, mode="human"):
#         return None
# 
#     def close(self):
#         if self.viewer:
#             self.viewer.close()
#             self.viewer = None
# 











