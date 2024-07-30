import numpy as np

# x和y分别代表当前要判断是否位于设备节点安全范围内的边缘服务器的x和y坐标,PM_coordinate是设备节点坐标，num_PM是设备节点数目，delta_coverage是设备节点安全距离
def safe_distance_judgment(PM_coordinate, num_PM, delta_coverage, x, y):
    for j in range(num_PM):
        tmp = PM_coordinate[j, :]
        if (x - tmp[0])**2 + (y - tmp[1])**2 <= delta_coverage ** 2:
            return 1

    return 0
