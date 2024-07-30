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
bandwith_ES = 40 * np.power(10, 6)  # 每个ES可分配的带宽为40MHz （可能需要再考虑）
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




T_all_slot = 0                                         # 定义1所有时隙的时延总和
E_all_slot = 0                                         # 定义1所有时隙的能耗总和
M_coordinate_x = 23.0                                  # 假定M-IPD初始x坐标为23,初始y坐标可由AGV的运动轨迹y=x^0.9+1/10*x算出
v_M = 2                                                # M-IPD速度为2m/s (为x轴方向的分速度为2m/s)
T = 100                                                # 时隙数，需要再看论文确定
sigma_error = 0.2                                      # 由VAR模型的得到的误差






def evaluate_policy(args, env, agent, state_norm):
    times = 1
    evaluate_reward = 0
    for _ in range(times):
        s = env.reset()
        if args.use_state_norm:  # During the evaluating,update=False
            s = state_norm(s, update=False)
        done = False
        episode_reward = 0
        while not done:
            a = agent.evaluate(s)  # We use the deterministic policy during the evaluating
            s_, r, done, informa = env.step(a)
            if args.use_state_norm:
                s_ = state_norm(s_, update=False)
            episode_reward += r
            s = s_
        evaluate_reward += episode_reward

    return informa

def main(args, env_name, number, seed, location):  # M-IPD在ES1内时ES1的测试主函数
    global glo_M_x
    config.glo_M_x = location
    env_evaluate = gym.make(env_name)  # When evaluating the policy, we need to rebuild an environment
    # Set random seed
    env_evaluate.seed(seed)
    env_evaluate.action_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    args.state_dim = env_evaluate.observation_space.shape[0]
    args.action_dim = env_evaluate.action_space.n
    args.max_episode_steps = env_evaluate._max_episode_steps  # Maximum number of steps per episode

    state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization
    state_norm.running_ms.load('D:/yjs/journal/simulation/result/M_in_ES1/ES1/normalization/normalization_params_{}.npz'.format(location))
    agent = PPO_discrete(args)
    agent.load('D:/yjs/journal/simulation/result/M_in_ES1/ES1/actor_critic/ppo_model', location)  # 加载训练好的模型参数
    informa = evaluate_policy(args, env_evaluate, agent, state_norm)
    return informa

def main1(args, env_name, number, seed):  # M-IPD在ES1内时ES2的测试主函数

    env_evaluate = gym.make(env_name)  # When evaluating the policy, we need to rebuild an environment
    # Set random seed
    env_evaluate.seed(seed)
    env_evaluate.action_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    args.state_dim = env_evaluate.observation_space.shape[0]
    args.action_dim = env_evaluate.action_space.n
    args.max_episode_steps = env_evaluate._max_episode_steps  # Maximum number of steps per episode
    state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization
    state_norm.running_ms.load('D:/yjs/journal/simulation/result/M_in_ES1/ES2/normalization/normalization_params_{}.npz'.format(0))
    agent = PPO_discrete(args)
    agent.load('D:/yjs/journal/simulation/result/M_in_ES1/ES2/actor_critic/ppo_model', 0)  # 加载训练好的模型参数
    informa = evaluate_policy(args, env_evaluate, agent, state_norm)
    return informa

def main2(args, env_name, number, seed):  # M-IPD在ES2内时ES1的测试主函数

    env_evaluate = gym.make(env_name)  # When evaluating the policy, we need to rebuild an environment
    # Set random seed
    env_evaluate.seed(seed)
    env_evaluate.action_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    args.state_dim = env_evaluate.observation_space.shape[0]
    args.action_dim = env_evaluate.action_space.n
    args.max_episode_steps = env_evaluate._max_episode_steps  # Maximum number of steps per episode
    state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization
    state_norm.running_ms.load('D:/yjs/journal/simulation/result/M_in_ES2/ES1/normalization/normalization_params_{}.npz'.format(0))
    agent = PPO_discrete(args)
    agent.load('D:/yjs/journal/simulation/result/M_in_ES2/ES1/actor_critic/ppo_model', 0)  # 加载训练好的模型参数
    informa = evaluate_policy(args, env_evaluate, agent, state_norm)
    return informa

def main3(args, env_name, number, seed, location):  # M-IPD在ES2内时ES2的测试主函数
    global glo_M_x
    config.glo_M_x = location
    env_evaluate = gym.make(env_name)  # When evaluating the policy, we need to rebuild an environment
    # Set random seed
    env_evaluate.seed(seed)
    env_evaluate.action_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    args.state_dim = env_evaluate.observation_space.shape[0]
    args.action_dim = env_evaluate.action_space.n
    args.max_episode_steps = env_evaluate._max_episode_steps  # Maximum number of steps per episode
    state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization
    state_norm.running_ms.load('D:/yjs/journal/simulation/result/M_in_ES2/ES2/normalization/normalization_params_{}.npz'.format(location))
    agent = PPO_discrete(args)
    agent.load('D:/yjs/journal/simulation/result/M_in_ES2/ES2/actor_critic/ppo_model', location)  # 加载训练好的模型参数
    informa = evaluate_policy(args, env_evaluate, agent, state_norm)
    return informa

#########################################################################可调整parser参数
parser = argparse.ArgumentParser("Hyperparameter Setting for PPO-discrete")
parser.add_argument("--max_train_steps", type=int, default=int(8e5), help=" Maximum number of training steps")
parser.add_argument("--evaluate_freq", type=float, default=5e3, help="Evaluate the policy every 'evaluate_freq' steps")
parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
parser.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")
parser.add_argument("--hidden_width", type=int, default=64, help="The number of neurons in hidden layers of the neural network")
parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")
parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
parser.add_argument("--use_state_norm", type=bool, default=True, help="Trick 2:state normalization")
parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")
args = parser.parse_args()
############################################################################################

A = 120                                               # 将时延单位标准化为标准成本单位的值
B = 90                                                 # 将能耗单位标准化为标准成本单位的值
w_1 = 0.5                                              # 时延成本对应的权重
w_2 = 0.9                                              # 能耗成本对应的权重
T_run_1 = 52  # M-IPD在ES1内的运动时隙数
arr_ES1 = []
arr_ES2 = []
'在所有时隙都进行资源的重新分配'
# a = np.zeros([T, 1], dtype=np.float32)
# Delay_ES = np.zeros([T, 1], dtype=np.float32)                                         # 定义所有时隙的总时延
# Energy_ES = np.zeros([T, 1], dtype=np.float32)                                        # 定义所有时隙的总能耗（不包括重新分配资源时的计算能耗）
# cost = np.zeros([T, 1], dtype=np.float32)                                               # 定义所有时隙的时延能耗成本（不包括重新分配资源时的计算能耗）
# for t in range(T_run_1):
#     M_x = M_coordinate_x + v_M * t * (50 / T)  # 当前时隙M-IPD的位置
#     information_ES_1 = main(args, env_name="MyEnv-v0", number=1, seed=0, location=M_x)       # 确定策略运行ES1范围内的资源分配信息
#     information_ES_2 = main1(args, env_name="MyEnv-v1", number=1, seed=0)                    # 确定策略运行ES2范围内的资源分配信息
#     Delay_ES[t] = information_ES_1['T'][0] + information_ES_2['T'][0]                      # 当前时隙下的ES1内和ES2内的时延和
#     Energy_ES[t] = information_ES_1['T'][1] + information_ES_2['T'][1]                     # 当前时隙下的ES1和ES2内的能耗和（不包括重新分配资源时的计算能耗）
#     arr_ES1.append(information_ES_1['T'][2])
#     arr_ES2.append(information_ES_2['T'][2])
#     Energy_ES[t] += 1                                                                      # ES1每进行一次资源的重新分配都需要增加一个ES的计算能耗
# for t in range(T_run_1, (T - 2)):
#     M_x = M_coordinate_x + v_M * t * (50 / T)  # 当前时隙M-IPD的位置
#     information_ES_2 = main3(args, env_name="MyEnv-v3", number=1, seed=0, location=M_x)      # 确定策略运行ES2范围内的资源分配信息
#     information_ES_1 = main2(args, env_name="MyEnv-v2", number=1, seed=0)                    # 确定策略运行ES1范围内的资源分配信息
#     Delay_ES[t] = information_ES_2['T'][0] + information_ES_1['T'][0]                        # ES1内当前时隙下的时延
#     Energy_ES[t] = information_ES_2['T'][1] + information_ES_1['T'][1]                       # ES1内当前时隙下的能耗（不包括重新分配资源时的计算能耗）
#     arr_ES1.append(information_ES_1['T'][2])
#     arr_ES2.append(information_ES_2['T'][2])
#     Energy_ES[t] += 1                                                                        # ES1每进行一次资源的重新分配都需要增加一个ES的计算能耗
# Delay_ES[98] = 1.0950
# Delay_ES[99] = 1.1233
# Energy_ES[98] = 2.05423
# Energy_ES[99] = 2.05511
# cost = w_1 * A * Delay_ES + (1 - w_1) * B * Energy_ES
# print(sum(cost) / T)
# print((w_1 * 1 * sum(Delay_ES)) / T)
# print(((1 - w_1) * 1 * sum(Energy_ES)) / T)
#
# arr_ES1.append([5, 6, 6, 5, 5, 2, 5, 5, 5, 6])
# arr_ES1.append([5, 6, 6, 5, 5, 2, 5, 5, 5, 6])
# arr_ES2.append([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 25])
# arr_ES2.append([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 25])
# np_arr_ES1 = np.array(arr_ES1,dtype=object)
# np_arr_ES2 = np.array(arr_ES2, dtype=object)
# np.save('np_arr_ES1.npy', np_arr_ES1)
# np.save('np_arr_ES2.npy', np_arr_ES2)
'决策变量指导下的资源的重新分配'
num_122 = 0
threshold = 1.9
t_last = 0                                                                                           # 上次重新分配时间，初始值为0
M_x_last = 23.0                                                                                      # 上次重分配时M-IPDx坐标，初始为M-IPD的初始坐标
Delay_ES_Gamma = np.zeros([T, 1], dtype=np.float32)                                                  # 定义决策变量指导下的所有时隙的总时延
Energy_ES_Gamma = np.zeros([T, 1], dtype=np.float32)                                                 # 定义决策变量指导下的所有时隙的总能耗（不包括重新分配资源时的计算能耗）
for t in range(T_run_1):                                                                             # M-IPD在ES1内运动
    if t == 0:                                                                                       # 初始状态位置进行一次带宽的分配
        t_last = 0                                                                                   # 上次重新分配时间，初始值为0
        M_x_last = 23.0                                                                              # 上次重分配时M-IPDx坐标，初始为M-IPD的初始坐标
        information_ES_1_Gamma = main(args, env_name="MyEnv-v0", number=1, seed=0, location = 23.0)  # 当M-IPD在ES1范围内时确定策略运行ES1范围内的资源分配信息
        information_ES_2_Gamma = main1(args, env_name="MyEnv-v1", number=1, seed=0)                  # 当M-IPD在ES1范围内时确定策略运行ES2范围内的资源分配信息
        bandwidth_last = information_ES_1_Gamma['T'][2]                                              # 获得上次的带宽分配方案


    delta_psi = t - t_last                                                                                                        # 当前带宽占用时间
    d_last = mt.sqrt(mt.pow(M_x_last - ES_coordinate_data[0][0], 2) + mt.pow(AGV_trace(M_x_last) - ES_coordinate_data[0][1], 2))  # 上次重分配时M-IPD和ES1之间的距离
    M_x = M_coordinate_x + v_M * t * (50 / T)                                                                                     # 当前时隙M-IPD的位置
    d_now = mt.sqrt(mt.pow(M_x - ES_coordinate_data[0][0], 2) + mt.pow(AGV_trace(M_x) - ES_coordinate_data[0][1], 2))             #  当前M-IPD和ES1之间的距离
    delta_d = abs(d_now - d_last)                                                                                                 # 当前距离变化量
    delta_psi = (2 / mt.pi) * mt.atan(delta_psi)                                                                                  # 对带宽占用时间变量进行归一化
    delta_d = (2 / mt.pi) * mt.atan(delta_d)                                                                                      # 对距离变量进行归一化
    Gamma = np.log2( (1 + delta_psi) * (1 +  delta_d) )                                                                            # 重分配决策变量值计算

    flag_band = 0                                                                                   # 判断M-IPD的所需要带宽是否大于上次分配，0为没有大于，1为大于，需要进行重新分配
    information_test = main(args, env_name="MyEnv-v0", number=1, seed=0, location = M_x)            # 判断当前时刻M-IPD所需要的带宽是否大于上次所分配的带宽，若是大于，则也需要进行带宽的重新分配
    bandwidth_test = information_test['T'][2]
    if bandwidth_test[10] > bandwidth_last[10]:
        flag_band = 1
    else:
        flag_band = 0

    if Gamma >= threshold or flag_band == 1:      # 当决策变量的值大于所设定的阈值时需要进行资源的重新分配
        num_122+= 1
        information_ES_1_Gamma = main(args, env_name="MyEnv-v0", number=1, seed=0, location = M_x)  # 当M-IPD在ES1范围内时确定策略运行ES1范围内的资源分配信息
        Delay_ES_Gamma[t] = information_ES_1_Gamma['T'][0] + information_ES_2_Gamma['T'][0]         # 当前时隙下的ES1内和ES2内的时延和
        Energy_ES_Gamma[t] = information_ES_1_Gamma['T'][1] + information_ES_2_Gamma['T'][1]        # ES1内当前时隙下的能耗（不包括重新分配资源时的计算能耗）
        Energy_ES_Gamma[t] += 1                                                                     # ES1每进行一次资源的重新分配都需要增加一个ES的计算能耗
        t_last = t                                                                                  # 将当前时隙赋给t_last
        M_x_last = M_x                                                                              # 将当前M-IPD位置赋给M_x_last
        bandwidth_last = information_ES_1_Gamma['T'][2]                                             # 将当前带宽分配方案赋给bandwidth_last
    else:                                                                                           # 没有达到所设定的阈值不需要进行带宽的重新分配
        E_trans = 0  # 定义传输能耗
        T_trans = Delay_with_M_IPD(data_trans, bandwidth_last, ES_coordinate_data[0], 10, IPD_coordinate_1, M_x)
        Delay_ES_Gamma[t] = sum(T_trans) + information_ES_2_Gamma['T'][0]
        for i in range(10):
            E_trans += ((10 ** (power_tx_F / 10)) / 1000) * T_trans[i]  # F-IPDs的传输能耗和
        E_trans += ((10 ** (power_tx_M / 10)) / 1000) * T_trans[10] + ((10 ** (power_tx_ES / 10)) / 1000) * ( data_ES / R_ES_to_SO)
        Energy_ES_Gamma[t] = E_trans + p_SO_cycle * CPU_cn + information_ES_2_Gamma['T'][1]        # ES1内当前时隙下的能耗，没有重新分配的计算能耗


for t in range(T_run_1, (T - 2)):
    if t == T_run_1:                                                                                # M-IPD在ES2内运动
        t_last = T_run_1                                                                             # 上次重新分配时间，此时为T_run_1
        M_x_last = 75.0                                                                              # 上次重分配时M-IPDx坐标，此时为75
        information_ES_2_Gamma = main3(args, env_name="MyEnv-v3", number=1, seed=0, location = 75.0)  # 当M-IPD在ES2范围内时确定策略运行ES2范围内的资源分配信息
        information_ES_1_Gamma = main2(args, env_name="MyEnv-v2", number=1, seed=0)                  # 当M-IPD在ES2范围内时确定策略运行ES1范围内的资源分配信息
        bandwidth_last = information_ES_2_Gamma['T'][2]                                              # 获得上次的带宽分配方案

    delta_psi = t - t_last                                                                                                        # 当前带宽占用时间
    d_last = mt.sqrt(mt.pow(M_x_last - ES_coordinate_data[1][0], 2) + mt.pow(AGV_trace(M_x_last) - ES_coordinate_data[1][1], 2))  # 上次重分配时M-IPD和ES1之间的距离
    M_x = M_coordinate_x + v_M * t * (50 / T)                                                                                     # 当前时隙M-IPD的位置
    d_now = mt.sqrt(mt.pow(M_x - ES_coordinate_data[1][0], 2) + mt.pow(AGV_trace(M_x) - ES_coordinate_data[1][1], 2))             #  当前M-IPD和ES1之间的距离
    delta_d = abs(d_now - d_last)                                                                                                 # 当前距离变化量
    delta_psi = (2 / mt.pi) * mt.atan(delta_psi)                                                                                  # 对带宽占用时间变量进行归一化
    delta_d = (2 / mt.pi) * mt.atan(delta_d)                                                                                      # 对距离变量进行归一化
    Gamma = np.log2( (1 + delta_psi) * (1 +  delta_d) )                                                                            # 重分配决策变量值计算

    flag_band = 0                                                                                   # 判断M-IPD的所需要带宽是否大于上次分配，0为没有大于，1为大于，需要进行重新分配
    information_test = main3(args, env_name="MyEnv-v3", number=1, seed=0, location = M_x)            # 判断当前时刻M-IPD所需要的带宽是否大于上次所分配的带宽，若是大于，则也需要进行带宽的重新分配
    bandwidth_test = information_test['T'][2]
    if bandwidth_test[12] > bandwidth_last[12]:
        flag_band = 1
    else:
        flag_band = 0

    if Gamma >= threshold or flag_band == 1:                                                              # 当决策变量的值大于所设定的阈值时需要进行资源的重新分配
        num_122+= 1
        information_ES_2_Gamma = main3(args, env_name="MyEnv-v3", number=1, seed=0, location = M_x)  # 当M-IPD在ES1范围内时确定策略运行ES1范围内的资源分配信息
        Delay_ES_Gamma[t] = information_ES_2_Gamma['T'][0] + information_ES_1_Gamma['T'][0]         # 当前时隙下的ES1内和ES2内的时延和
        Energy_ES_Gamma[t] = information_ES_2_Gamma['T'][1] + information_ES_1_Gamma['T'][1]        # ES1内当前时隙下的能耗（不包括重新分配资源时的计算能耗）
        Energy_ES_Gamma[t] += 1                                                                     # ES1每进行一次资源的重新分配都需要增加一个ES的计算能耗
        t_last = t                                                                                  # 将当前时隙赋给t_last
        M_x_last = M_x                                                                              # 将当前M-IPD位置赋给M_x_last
        bandwidth_last = information_ES_2_Gamma['T'][2]                                             # 将当前带宽分配方案赋给bandwidth_last
    else:                                                                                           # 没有达到所设定的阈值不需要进行带宽的重新分配
        E_trans = 0  # 定义传输能耗
        T_trans = Delay_with_M_IPD(data_trans, bandwidth_last, ES_coordinate_data[1], 12, IPD_coordinate_2, M_x)
        Delay_ES_Gamma[t] = sum(T_trans) + information_ES_1_Gamma['T'][0]         # 当前时隙下的ES1内和ES2内的时延和
        for i in range(12):
            E_trans += ((10 ** (power_tx_F / 10)) / 1000) * T_trans[i]  # F-IPDs的传输能耗和
        E_trans += ((10 ** (power_tx_M / 10)) / 1000) * T_trans[12] + ((10 ** (power_tx_ES / 10)) / 1000) * ( data_ES / R_ES_to_SO)
        Energy_ES_Gamma[t] = E_trans + p_SO_cycle * CPU_cn + information_ES_1_Gamma['T'][1]        # ES1内当前时隙下的能耗，没有重新分配的计算能耗
Delay_ES_Gamma[98] = 1.0950
Delay_ES_Gamma[99] = 1.1233
Energy_ES_Gamma[98] = 2.05423
Energy_ES_Gamma[99] = 2.05511
cost = w_1 * A * Delay_ES_Gamma + (1 - w_1) * B * Energy_ES_Gamma
print(sum(cost) / T)
print((w_1 * 1* sum(Delay_ES_Gamma)) / T)
print(((1 - w_1) * 1 * sum(Energy_ES_Gamma)) / T)
print(num_122)







