import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# 将数据增加一个维度，最后一位是1
def augment(xys):
    axy = np.ones((len(xys), len(xys[0]) + 1))
    axy[:, :len(xys[0])] = xys
    return axy

# 计算方程组的解
def estimate(xys):
    axy = augment(xys)
    return np.linalg.svd(axy)[-1][-1, :]

# 判断是否是inlier点， 方程：ax + by + c < threshold
def is_inlier(coeffs, xy, threshold = 0.01):
    return np.abs(coeffs.dot(augment([xy]).T)) < threshold

def run_ransac(data, sample_size, goal_inliers, max_iterations, stop_at_goal=True, random_seed=None):
    """
    :param data: 待拟合数据
    :param sample_size:
            2：两个点确定一条线
            3：三个点确定一个平面。
    :param goal_inliers: inliers点的个数
    :param max_iterations: 最大迭代次数
    :param stop_at_goal: 是否在满足goal_inliers条件时候结束迭代
    :param random_seed: 随机初始化种子
    :return:
            1：返回拟合参数（a, b, c ...）
            2: 拟合数据量
    """
    best_ic = 0
    best_model = None
    random.seed(random_seed)
    # random.sample cannot deal with "data" being a numpy array
    data = list(data)
    for i in range(max_iterations):
        s = random.sample(data, int(sample_size))
        m = estimate(s)
        ic = 0
        for j in range(len(data)):
            if is_inlier(m, data[j]):
                ic += 1
        if ic > best_ic:
            best_ic = ic
            best_model = m
            if ic > goal_inliers and stop_at_goal:
                break
    print('took iterations:', i+1, 'best model:', best_model, 'explains:', best_ic)
    return best_model, best_ic

def line_fit():
    n = 100
    max_iterations = 100
    goal_inliers = n * 0.3
    # test data
    xys = np.random.random((n, 2)) * 10
    xys[:50, 1:] = xys[:50, :1]
    plt.scatter(xys.T[0], xys.T[1])

    # RANSAC
    m, b = run_ransac(xys, 2, goal_inliers, max_iterations)
    a, b, c = m

    plt.plot([0, 10], [-c / b, -(c + 10 * a) / b], color=(0, 1, 0))
    plt.show()

def plane_fit():
    fig = plt.figure()
    ax = mplot3d.Axes3D(fig)

    def plot_plane(a, b, c, d):
        xx, yy = np.mgrid[:10, :10]
        return xx, yy, (-d - a * xx - b * yy) / c

    n = 100
    max_iterations = 100
    goal_inliers = n * 0.3

    # test data
    xyzs = np.random.random((n, 3)) * 10
    xyzs[:50, 2:] = xyzs[:50, :1]

    ax.scatter3D(xyzs.T[0], xyzs.T[1], xyzs.T[2])

    # RANSAC
    m, b = run_ransac(xyzs, 3, goal_inliers, max_iterations)
    a, b, c, d = m
    xx, yy, zz = plot_plane(a, b, c, d)
    ax.plot_surface(xx, yy, zz, color=(0, 1, 0, 0.5))

    plt.show()
if __name__ == '__main__':
    # line_fit()  # 拟合直线
    plane_fit()  # 拟合平面
