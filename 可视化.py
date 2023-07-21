import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
# 定义全局变量



N = 80  # 网格数
h = 2.0 / (N - 1)  # 网格宽度
dt = 0.9 * 2.0 / 7 * h  # 时间步长
tn = round(10 / dt)  # 时间步数
origin = 0  # 原点

x = np.linspace(-1.0, 1.0, N)  # 生成0到1之间长度为N的等差数列
x1 = x + h / 2.0

u2 = np.zeros((N, tn))  # 理论值
t = np.linspace(0, tn * dt, tn + 1)

# 计算理论解
for i in range(tn):
    for j in range(N):
        option = i % 2
        if option == 0:
            u2[j, i] = np.sin(np.pi * (x[j] - t[i]))
        else:
            u2[j, i] = np.sin(np.pi * (x1[j] - t[i]))



np.savetxt("u2.csv", u2, delimiter=",")  # 保存为csv文件
u = np.loadtxt("u.csv", delimiter=",")



fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

# 定义更新函数
def update(i):
   
    option = i % 2
    # 清空当前图形
    ax1.clear()
    ax2.clear()
    plt.xlim(-1, 1);
    # 根据选项绘制不同的图形
    if option == 1:
        ax1.plot(x1, u2[:,i])
        ax2.plot(x1, u2[:,i]-u[:,i],"--")
    elif option == 0:
        ax1.plot(x, u2[:,i])
        ax2.plot(x, u2[:,i]-u[:,i],"--")
    elif option == 3:
        ax1.plot(x, u2[:,i])
    

# 创建动画
ani = FuncAnimation(fig, update, interval=1)

# 显示动画
plt.show()
