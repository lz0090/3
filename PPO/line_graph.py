import matplotlib.pyplot as plt
import numpy as np
'带宽变化对成本的影响'
plt.rcParams['font.family'] = 'Times New Roman'
x =[30, 40, 50, 60, 70]

system_cost_DQN = [192.61641, 169.16278, 154.82687, 145.12572, 138.1089]
delay_cost_DQN = [99.98022, 76.69886, 62.468224, 52.838345, 45.873024]
energy_cost_DQN = [92.63626, 92.46396, 92.35855, 92.28739, 92.23581]

# 绘制折线图
plt.plot(x, system_cost_DQN, 's-', color='red', label='DQN')


# 添加标题和坐标轴标签
plt.xlabel('Allocatable bandwidth per ES', fontsize=16)
plt.ylabel('Delay-Energy cost', fontsize=16)
# 调整刻度标签的字号
plt.xticks(x, fontsize=16)
plt.yticks(fontsize=16)
plt.ylim(0, 100)
# 设置图例，并调整图例字号
plt.legend(loc='upper right', fontsize=16)

# 显示网格
plt.grid(True)

# 显示图表
plt.savefig('D:/桌面/2.pdf', bbox_inches='tight')
