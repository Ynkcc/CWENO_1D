import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
# 设置网格尺寸、时间步长、计算时间范围
dx = dy = 0.1
dt = 0.01
T = 2.0

# 设置计算网格尺寸、时间步数
nx = ny = 201
nt = int(T / dt)

# 设置波速和初始条件
c = 1.0
u = np.zeros((nx, ny))
u_old = np.zeros((nx, ny))
for i in range(nx):
    for j in range(ny):
        u[i,j] = np.sin(np.pi * i * dx) * np.sin(np.pi * j * dy)

# 设置弹性边界条件
def set_bound(u):
    for i in range(nx):
        u[i,0] = u_old[i,1]
        u[i,-1] = u_old[i,-2]
    for j in range(ny):
        u[0,j] = u_old[1,j]
        u[-1,j] = u_old[-2,j]

# 初始化动画图像
fig = plt.figure()
ax = fig.add_axes(Axes3D(fig))
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u')


surf = None  # 全局范围内声明surf对象

x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)
X, Y = np.meshgrid(x, y)
ax.plot_surface(X, Y, u, cmap='coolwarm')


# 定义更新函数
def update(frame):
    global u, u_old, surf,X,Y
    u_old[:] = u[:]
    set_bound(u)
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            u[i,j] = 2*u_old[i,j] - u[i,j] + c**2 * dt**2 / dx**2 * (u_old[i+1,j] - 2*u_old[i,j] + u_old[i-1,j]) \
                                                 + c**2 * dt**2 / dy**2 * (u_old[i,j+1] - 2*u_old[i,j] + u_old[i,j-1])
    ax.clear()
    ax.set_xlim3d(0, 2)
    ax.set_ylim3d(0, 2)
    ax.set_zlim3d(-1, 1)
    if surf is not None:  # 如果surf对象已经存在，则更新它，否则创建新的surf对象
        surf.remove()
    surf = ax.plot_surface(X, Y, u, cmap='coolwarm')

    return surf,

# 生成动画

ani = FuncAnimation(fig, update, frames=nt, blit=True)
plt.show()
