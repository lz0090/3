import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Times New Roman'
A = 100
B = 70
# 数据集
x = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35] # 2,1.95,1.9,1.7,1.55,1.45,1.3
Delay_all = [0.47081738, 0.4564017, 0.44302784, 0.43879105, 0.43827694,0.43810497,  0.4380034]  # 第一组数据
Energy_all = [0.5710842, 0.63098507, 0.6908882, 0.800861, 0.8708573,0.9908562, 1.01085526]   # 第二组数据，将被堆叠在y1之上
difference_Delay_all = np.diff(Delay_all)
y1 = [0] * len(x)
y1[6] = Delay_all[6] * A
y1[5] = y1[6] - 10000 * difference_Delay_all[5]
y1[4] = y1[5] - 10000 * difference_Delay_all[4]
y1[3] = y1[4] - 3000 * difference_Delay_all[3]
y1[2] = y1[3] - 1500 * difference_Delay_all[2]
y1[1] = y1[2] - 500 * difference_Delay_all[1]
y1[0] = y1[1] - 500 * difference_Delay_all[0]
y2 = B * np.array(Energy_all)

# 创建堆叠柱状图
plt.bar(x, y1, label='Delay cost', width=0.03)  # 第一组数据
plt.bar(x, y2, label='Energy cost', width=0.03, bottom=y1)  # 第二组数据，指定bottom参数为y1堆叠在其上

# 添加标题和标签，并调整字号
plt.xlabel('Prediction error', fontsize=16)
plt.ylabel('Delay-Energy cost', fontsize=16)

# 调整刻度标签的字号
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# 设置y轴的范围
plt.ylim(0, 150)

# 添加一条y = 115.602946的横虚线
dashed_line_value = 115.602946
plt.axhline(y=dashed_line_value, color='r', linestyle='--', linewidth=2, label='Target cost')

# 设置图例，并调整图例字号
# plt.legend(loc='upper left', fontsize=16)

# 显示图表
plt.show()
# plt.savefig('D:/桌面/1.pdf', bbox_inches='tight')