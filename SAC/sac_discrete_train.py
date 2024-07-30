import gym
import torch
import numpy as np
import config_MIPD
import matplotlib.pyplot as plt
from sac_discrete import ReplayBuffer, SAC
import random

seed_value = 0
torch.manual_seed(seed_value)
np.random.seed(seed_value)
random.seed(seed_value)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_value)

# -------------------------------------- #
# 参数设置
# -------------------------------------- #

num_epochs = 200  # 训练回合数
capacity = 10000  # 经验池容量
min_size = 500 # 经验池训练容量
batch_size = 64
n_hiddens = 128
actor_lr = 1e-3  # 策略网络学习率
critic_lr = 1e-2  # 价值网络学习率
alpha_lr = 1e-2  # 熵权重的学习率
target_entropy = -1
tau = 0.005  # 软更新参数
gamma = 0.98  # 折扣因子
device = torch.device('cuda') if torch.cuda.is_available()else torch.device('cpu')

# -------------------------------------- #
# 环境加载
# -------------------------------------- #

env_name = "MyEnv-v0"
env = gym.make(env_name)
env.seed(seed_value)  # 设置环境的随机种子
n_states = env.observation_space.shape[0]  # 状态数
n_actions = env.action_space.n  # 动作数

# -------------------------------------- #
# 模型构建
# -------------------------------------- #

agent = SAC(n_states = n_states,
            n_hiddens = n_hiddens,
            n_actions = n_actions,
            actor_lr = actor_lr,
            critic_lr = critic_lr,
            alpha_lr = alpha_lr,
            target_entropy = target_entropy,
            tau = tau,
            gamma = gamma,
            device = device,
            )

# -------------------------------------- #
# 经验回放池
# -------------------------------------- #

buffer = ReplayBuffer(capacity=capacity)

# -------------------------------------- #
# 模型构建
# -------------------------------------- #

return_list = []  # 保存每回合的return
global glo_M_x  # 定义M-IPD的x坐标为全局变量
config_MIPD.glo_M_x = 23.0  # 导入全局变量值

for i in range(num_epochs):
    state = env.reset()
    epochs_return = 0  # 累计每个时刻的reward
    done = False  # 回合结束标志

    while not done:
        # 动作选择
        action = agent.take_action(state)

        # 环境更新
        next_state, reward, done, _ = env.step(action)
        # 将数据添加到经验池
        buffer.add(state, action, reward, next_state, done)
        # 状态更新
        state = next_state
        # 累计回合奖励
        epochs_return += reward

        # 经验池超过要求容量，就开始训练
        if buffer.size() > min_size:
            s, a, r, ns, d = buffer.sample(batch_size)  # 每次取出batch组数据
            # 构造数据集
            transition_dict = {'states': s,
                               'actions': a,
                               'rewards': r,
                               'next_states': ns,
                               'dones': d}
            # 模型训练
            agent.update(transition_dict)
    # 保存每个回合return
    return_list.append(epochs_return)

    # 打印回合信息
    print(f'iter:{i}, return:{np.mean(return_list[-10:])}')

# -------------------------------------- #
# 绘图
# -------------------------------------- #

plt.plot(return_list)
plt.title('return')
plt.show()