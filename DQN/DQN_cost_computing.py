import numpy as np
import random
import math as mt
import matplotlib.pyplot as plt
import gym
from AGVTrace import AGV_trace
import torch
from torch import nn
from torch import optim
import matplotlib.pyplot as plt
from collections import deque, namedtuple  # 队列类型
import config

R_ES_to_SO = 15 * np.power(10, 6)        # 根据‘Multitask Multiobjective Deep Reinforcement Learning-Based Computation Offloading Method for Industrial Internet of Things’设置的ES到SO层的传输速率为15Mbits/s
data_ES = 0.1 * np.power(10, 6)          # 设ES传到SO层的数据量为0.1M，因为只是关键数据，所以应该比较小
CPU_cn = 1                               # ‘Multitask Multiobjective Deep Reinforcement Learning-Based Computation Offloading Method for Industrial Internet of Things’中设的每个任务需要的CPU周期为[0.8,8.4]G/task，这里取1G表示SO层计算所需
f_SO_CPU = 30                            # ‘Multitask Multiobjective Deep Reinforcement Learning-Based Computation Offloading Method for Industrial Internet of Things’中设的每个任务需要的CPU周期为[20,40]GHz，这里取30GHz为SO层的计算能力
p_SO_cycle = 0.5                         # 每个CPU周期能耗，这个数值大小还要根据最后的仿真数值来调整
data_trans = 2 * np.power(10, 6)    # 上行传输的数据量为2M
bandwith_ES = 40 * np.power(10, 6)  # 每个ES可分配的带宽为40MHz （可能需要再考虑）
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


T_all_slot = 0                                         # 定义1所有时隙的时延总和
E_all_slot = 0                                         # 定义1所有时隙的能耗总和
M_coordinate_x = 23.0                                  # 假定M-IPD初始x坐标为23,初始y坐标可由AGV的运动轨迹y=x^0.9+1/10*x算出
v_M = 2                                                # M-IPD速度为2m/s (为x轴方向的分速度为2m/s)
T = 100                                                # 时隙数，需要再看论文确定

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayMemory(object):

    def __init__(self, memory_size):
        self.memory = deque([], maxlen=memory_size)

    def sample(self, batch_size):
        batch_data = random.sample(self.memory, batch_size)
        state, action, reward, next_state, done = zip(*batch_data)
        return state, action, reward, next_state, done

    def push(self, *args):
        # *args: 把传进来的所有参数都打包起来生成元组形式
        # self.push(1, 2, 3, 4, 5)
        # args = (1, 2, 3, 4, 5)
        self.memory.append(Transition(*args))

    def __len__(self):
        return len(self.memory)
class Qnet(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(Qnet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(n_observations, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, state):
        return self.model(state)
class Agent(object):

    def save(self, filepath, M_x):
        torch.save(self.q_net.state_dict(), filepath + '_' + str(M_x) + '_q_net.pth')
        torch.save(self.target_q_net.state_dict(), filepath + '_' + str(M_x) + '_target_q_net.pth')

    def load(self, filepath, M_x):
        self.q_net.load_state_dict(torch.load(filepath + '_' + str(M_x) + '_q_net.pth'))
        self.target_q_net.load_state_dict(torch.load(filepath + '_' + str(M_x) + '_target_q_net.pth'))

    def __init__(self, observation_dim, action_dim, gamma, lr, epsilon, target_update):
        self.action_dim = action_dim
        self.q_net = Qnet(observation_dim, action_dim).to(device)
        self.target_q_net = Qnet(observation_dim, action_dim).to(device)
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        self.target_update = target_update
        self.count = 0

        self.optimizer = optim.Adam(params=self.q_net.parameters(), lr=lr)
        self.loss = nn.MSELoss()

    def take_action(self, state):
        if np.random.uniform(0, 1) < 1 - self.epsilon:
            state = torch.tensor(state, dtype=torch.float).to(device)
            action = torch.argmax(self.q_net(state)).item()
        else:
            action = np.random.choice(self.action_dim)
        return action

    def update(self, transition_dict):

        states = transition_dict.state
        actions = np.expand_dims(transition_dict.action, axis=-1)  # 扩充维度
        rewards = np.expand_dims(transition_dict.reward, axis=-1)  # 扩充维度
        next_states = transition_dict.next_state
        dones = np.expand_dims(transition_dict.done, axis=-1)  # 扩充维度

        states = torch.tensor(states, dtype=torch.float).to(device)
        actions = torch.tensor(actions, dtype=torch.int64).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float).to(device)
        dones = torch.tensor(dones, dtype=torch.float).to(device)
        # update q_values
        # gather(1, acitons)意思是dim=1按行号索引， index=actions
        # actions=[[1, 2], [0, 1]] 意思是索引出[[第一行第2个元素， 第1行第3个元素],[第2行第1个元素， 第2行第2个元素]]
        # 相反，如果是这样
        # gather(0, acitons)意思是dim=0按列号索引， index=actions
        # actions=[[1, 2], [0, 1]] 意思是索引出[[第一列第2个元素， 第2列第3个元素],[第1列第1个元素， 第2列第2个元素]]
        # states.shape(64, 4) actions.shape(64, 1), 每一行是一个样本，所以这里用dim=1很合适
        predict_q_values = self.q_net(states).gather(1, actions)
        with torch.no_grad():
            # max(1) 即 max(dim=1)在行向找最大值，这样的话shape(64, ), 所以再加一个view(-1, 1)扩增至(64, 1)
            max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
            q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)
        l = self.loss(predict_q_values, q_targets)

        self.optimizer.zero_grad()
        l.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            # copy model parameters
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.count += 1
def run_episode(env, agent, repalymemory, batch_size):
    state = env.reset()
    reward_total = 0
    while True:
        action = agent.take_action(state)
        next_state, reward, done, _ = env.step(action)
        # print(reward)
        repalymemory.push(state, action, reward, next_state, done)
        reward_total += reward
        if len(repalymemory) > batch_size:
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = repalymemory.sample(batch_size)
            T_data = Transition(state_batch, action_batch, reward_batch, next_state_batch, done_batch)
            # print(T_data)
            agent.update(T_data)
        state = next_state
        if done:
            break
    return reward_total
def episode_evaluate(env, agent):
    reward_list = []
    for i in range(1):
        state = env.reset()
        done = False
        reward_episode = 0
        step = 0
        while not done:
            action = agent.take_action(state)
            step += 1
            next_state, reward, done, informa = env.step(action)
            reward_episode += reward
            state = next_state
        reward_list.append(reward_episode)
    return informa

def main(band_M, location):  # M-IPD在ES1内时ES1的测试主函数
    global glo_M_x
    config.glo_M_x = location
    config.glo_band_M = band_M
    env_name = "MyEnv-v0"
    env = gym.make(env_name)
    env.seed(0)
    env.action_space.seed(0)
    random.seed(0)
    np.random.seed(0)
    env.seed(0)
    torch.manual_seed(0)
    observation_n, action_n = env.observation_space.shape[0], env.action_space.n
    # print(observation_n, action_n)
    agent = Agent(observation_n, action_n, gamma=0.98, lr=2e-3, epsilon=0.01, target_update=10)
    agent.load('D:/yjs/journal/simulation/result/DQN/M_in_ES1/ES1/DQN_model', location)
    informa = episode_evaluate(env, agent)
    return informa

def main1(location):  # M-IPD在ES1内时ES2的测试主函数
    env_name = "MyEnv-v1"
    env = gym.make(env_name)
    env.seed(0)
    env.action_space.seed(0)
    random.seed(0)
    np.random.seed(0)
    env.seed(0)
    torch.manual_seed(0)
    observation_n, action_n = env.observation_space.shape[0], env.action_space.n
    # print(observation_n, action_n)
    agent = Agent(observation_n, action_n, gamma=0.98, lr=2e-3, epsilon=0.01, target_update=10)
    agent.load('D:/yjs/journal/simulation/result/DQN/M_in_ES1/ES2/DQN_model', location)
    informa = episode_evaluate(env, agent)
    return informa

def main2(location):  # M-IPD在ES2内时ES1的测试主函数
    env_name = "MyEnv-v2"
    env = gym.make(env_name)
    env.seed(0)
    env.action_space.seed(0)
    random.seed(0)
    np.random.seed(0)
    env.seed(0)
    torch.manual_seed(0)
    observation_n, action_n = env.observation_space.shape[0], env.action_space.n
    # print(observation_n, action_n)
    agent = Agent(observation_n, action_n, gamma=0.98, lr=2e-3, epsilon=0.01, target_update=10)
    agent.load('D:/yjs/journal/simulation/result/DQN/M_in_ES2/ES1/DQN_model', location)
    informa = episode_evaluate(env, agent)
    return informa

def main3(band_M, location):  # M-IPD在ES2内时ES1的测试主函数
    global glo_M_x
    config.glo_M_x = location
    config.glo_band_M = band_M
    env_name = "MyEnv-v3"
    env = gym.make(env_name)
    env.seed(0)
    env.action_space.seed(0)
    random.seed(0)
    np.random.seed(0)
    env.seed(0)
    torch.manual_seed(0)
    observation_n, action_n = env.observation_space.shape[0], env.action_space.n
    # print(observation_n, action_n)
    agent = Agent(observation_n, action_n, gamma=0.98, lr=2e-3, epsilon=0.01, target_update=10)
    agent.load('D:/yjs/journal/simulation/result/DQN/M_in_ES2/ES2/DQN_model', location)
    informa = episode_evaluate(env, agent)
    return informa

A = 120                                               # 将时延单位标准化为标准成本单位的值
B = 90                                                 # 将能耗单位标准化为标准成本单位的值
w_1 = 0.5                                              # 时延成本对应的权重
T_run_1 = 52  # M-IPD在ES1内的运动时隙数

'在所有时隙都进行资源的重新分配'
config.glo_band_all = 40 * np.power(10, 6)  # 每个ES可分配的总带宽
config.glo_data = 3 * np.power(10, 6)
all_M = np.load('./1.npy')  # 所有时隙给M-IPD分配的带宽
Delay_ES = np.zeros([T, 1], dtype=np.float32)                                         # 定义所有时隙的总时延
Energy_ES = np.zeros([T, 1], dtype=np.float32)                                        # 定义所有时隙的总能耗（不包括重新分配资源时的计算能耗）
cost = np.zeros([T, 1], dtype=np.float32)                                               # 定义所有时隙的时延能耗成本（不包括重新分配资源时的计算能耗）
for t in range(T_run_1):
    M_x = M_coordinate_x + v_M * t * (50 / T)  # 当前时隙M-IPD的位置
    information_ES_1 = main(band_M=int(all_M[t]), location=M_x)       # 确定策略运行ES1范围内的资源分配信息
    information_ES_2 = main1(location=0)                    # 确定策略运行ES2范围内的资源分配信息
    Delay_ES[t] = information_ES_1['T'][0] + information_ES_2['T'][0]                      # 当前时隙下的ES1内和ES2内的时延和
    Energy_ES[t] = information_ES_1['T'][1] + information_ES_2['T'][1]                     # 当前时隙下的ES1和ES2内的能耗和（不包括重新分配资源时的计算能耗）
    Energy_ES[t] += 1                                                                      # ES1每进行一次资源的重新分配都需要增加一个ES的计算能耗
for t in range(T_run_1, T):
    M_x = M_coordinate_x + v_M * t * (50 / T)  # 当前时隙M-IPD的位置
    information_ES_2 = main3(band_M=int(all_M[t]), location=M_x)      # 确定策略运行ES2范围内的资源分配信息
    information_ES_1 = main2(location=0)                  # 确定策略运行ES1范围内的资源分配信息
    Delay_ES[t] = information_ES_2['T'][0] + information_ES_1['T'][0]                        # ES1内当前时隙下的时延
    Energy_ES[t] = information_ES_2['T'][1] + information_ES_1['T'][1]                       # ES1内当前时隙下的能耗（不包括重新分配资源时的计算能耗）
    Energy_ES[t] += 1                                                                        # ES1每进行一次资源的重新分配都需要增加一个ES的计算能耗
cost = w_1 * A * Delay_ES + (1 - w_1) * B * Energy_ES
print(sum(cost) / T)
print((w_1 * A * sum(Delay_ES)) / T)
print(((1 - w_1) * B * sum(Energy_ES)) / T)