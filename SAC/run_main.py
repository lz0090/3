import gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from RL_brain import ReplayBuffer, SAC
import random
import config_MIPD
# -------------------------------------- #
# 参数设置
# -------------------------------------- #

def episode_evaluate(env, agent, M_x):
    global tmp_delay_last
    global tmp_energy_last
    global informa
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
        tmp_delay_now = informa['T'][0]
        tmp_energy_now = informa['T'][1]
        if tmp_delay_now <= tmp_delay_last and tmp_energy_now <= tmp_energy_last:
            tmp_delay_last = tmp_delay_now
            tmp_energy_last = tmp_energy_now
            agent.save_models('./train_model/M_in_ES1/ES1/SAC_model', M_x)

        reward_list.append(reward_episode)
    return informa


num_epochs = 50  # 训练回合数
capacity = 500  # 经验池容量
min_size = 200  # 经验池训练容量
batch_size = 64
n_hiddens = 64
actor_lr = 1e-3  # 策略网络学习率
critic_lr = 1e-2  # 价值网络学习率
alpha_lr = 1e-2  # 课训练变量的学习率
target_entropy = -1
tau = 0.005  # 软更新参数
gamma = 0.9  # 折扣因子
device = torch.device('cuda') if torch.cuda.is_available() \
    else torch.device('cpu')




# -------------------------------------- #
# 经验回放池
# -------------------------------------- #

buffer = ReplayBuffer(capacity=capacity)



# -------------------------------------- #
# 模型构建
# -------------------------------------- #

global glo_M_x  # 定义M-IPD的x坐标为全局变量
T = 100  # 暂定总时隙数为100
T_run = 52  # 在跑的时隙数，总时隙数是100的话就是20s，每个时隙0.5s
M_coordinate_x = 23  # M-IPD初始位置
v_M = 2  # M-IPD速度
all_M = np.load('./1.npy')  # 所有时隙给M-IPD分配的带宽
for t in range(T_run):

    return_list = []  # 保存每回合的return
    tmp_delay_last = 100
    tmp_energy_last = 100
    config_MIPD.glo_M_x = M_coordinate_x + v_M * t * (50 / T)  # 当前时隙M-IPD的位置
    config_MIPD.glo_band_M = int(all_M[t])

    env_name = "MyEnv-v0"
    env = gym.make(env_name)
    env.seed(0)
    env.action_space.seed(0)
    random.seed(0)
    np.random.seed(0)
    env.seed(0)
    torch.manual_seed(0)
    observation_n, action_n = env.observation_space.shape[0], env.action_space.n

    agent = SAC(n_states=observation_n,
                n_hiddens=n_hiddens,
                n_actions=action_n,
                actor_lr=actor_lr,
                critic_lr=critic_lr,
                alpha_lr=alpha_lr,
                target_entropy=target_entropy,
                tau=tau,
                gamma=gamma,
                device=device,
                )

    for i in range(num_epochs):
        state = env.reset()
        epochs_return = 0  # 累计每个时刻的reward
        done = False  # 回合结束标志
        while not done:
            # 动作选择
            action = agent.take_action(state)
            # 环境更新
            next_state, reward, done,_ = env.step(action)
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
            informa = episode_evaluate(env, agent, config_MIPD.glo_M_x)
            print(informa)


        # 保存每个回合return
        return_list.append(epochs_return)

        # 打印回合信息
        print(f'iter:{i}, return:{np.mean(return_list[-10:])}')
    print('-----------------------')




