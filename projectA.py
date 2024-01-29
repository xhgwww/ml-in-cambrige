
# 加载数据

import numpy as np

X = np.loadtxt('D:\\HuaweiMoveData\\Users\\hlc\\Desktop\\Preparing for Cambridge\\cam\\data_set_group_6\X.txt')
y = np.loadtxt('D:\\HuaweiMoveData\\Users\\hlc\\Desktop\\Preparing for Cambridge\\cam\\data_set_group_6\y.txt')

# 打乱

permutation = np.random.permutation(X.shape[0])  # 这样做的目的是打乱数据集的顺序，以提高模型的训练效果。
X = X[permutation, :]
y = y[permutation]

# 绘制数据

import matplotlib.pyplot as plt

##
# 绘制2D空间中点的函数，同时显示它们的标签
#
# 输入:
#
# X: 输入特征的2维数组
# y: 类别标签的1维数组（0或1）
#
# 输出: 显示在图中的点的x和y坐标矩阵


def plot_data_internal(X, y):  # 绘制散点图
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    plt.figure()
    plt.xlim(xx.min(None), xx.max(None))
    plt.ylim(yy.min(None), yy.max(None))
    ax = plt.gca()
    ax.plot(X[y == 0, 0], X[y == 0, 1], 'ro', label='类别 1')
    ax.plot(X[y == 1, 0], X[y == 1, 1], 'bo', label='类别 2')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('绘制数据')
    plt.legend(loc='upper left', scatterpoints=1, numpoints=1)
    return xx, yy

##
# 绘制数据，调用"plot_data_internal"函数，不返回任何内容。
#
# 输入:
#
# X: 输入特征的2维数组
# y: 类别标签的1维数组（0或1）
#
# 输出: 无
#

def plot_data(X, y):
    xx, yy = plot_data_internal(X, y)
    plt.show()

plot_data(X, y)

# 我们将数据分割成训练集和测试集

n_train = 800
X_train = X[0:n_train, :]
X_test = X[n_train:, :]
y_train = y[0:n_train]
y_test = y[n_train:]

# 逻辑函数

def logistic(x): return 1.0 / (1.0 + np.exp(-x))

##
# 使用逻辑分类器进行预测的函数
#
# 输入:
#
# X_tile: 输入特征的矩阵（在左侧附加了一个常数1） 
# w: 模型参数的向量
#
# 输出: 逻辑分类器的预测值
#

def predict(X_tilde, w): return logistic(np.dot(X_tilde, w))

##
# 计算逻辑分类器在某些数据上的平均对数似然
#
# 输入:
#
# X_tile: 输入特征的矩阵（在左侧附加了一个常数1） 
# y: 二进制输出标签的向量 
# w: 模型参数的向量
#
# 输出: 平均对数似然
#

def compute_average_ll(X_tilde, y, w):
    output_prob = predict(X_tilde, w)
    return np.mean(y * np.log(output_prob) + (1 - y) * np.log(1.0 - output_prob))

##
# 扩展输入特征矩阵，添加一个等于1的额外常数列。
#
# 输入:
#
# X: 输入特征的矩阵。
#
# 输出: 添加了额外常数列的特征矩阵 x_tilde。
#

def get_x_tilde(X): return np.concatenate((np.ones((X.shape[0], 1)), X), 1)

##
# 使用梯度下降优化似然来找到模型参数的函数
#
# 输入:
#
# X_tile_train: 训练输入特征的矩阵（在左侧附加了一个常数1） 
# y_train: 训练二进制输出标签的向量 
# X_tile_test: 测试输入特征的矩阵（在左侧附加了一个常数1） 
# y_test: 测试二进制输出标签的向量 
# alpha: 梯度下降优化的步长参数
# n_steps: 梯度下降优化的步数
#
# 输出: 
# 
# 1 - 模型参数向量 w 
# 2 - 在训练集上得到的平均对数似然值的向量
# 3 - 在测试集上得到的平均对数似然值的向量
#

def fit_w(X_tilde_train, y_train, X_tilde_test, y_test, n_steps, alpha):
    w = np.random.randn(X_tilde_train.shape[1])
    ll_train = np.zeros(n_steps)
    ll_test = np.zeros(n_steps)
    for i in range(n_steps):
        sigmoid_value = predict(X_tilde_train, w)

        # w = w - alpha * np.dot(X_tilde_train.T, sigmoid_value - y_train) / len(y_train) #用梯度下降训练
        # XXX 学生应完成的基于梯度的w更新规则

        gradient = np.dot((y_train-sigmoid_value).T, X_tilde_train)
        w = w + alpha * gradient

        ll_train[i] = compute_average_ll(X_tilde_train, y_train, w)
        ll_test[i] = compute_average_ll(X_tilde_test, y_test, w)
        print(ll_train[i], ll_test[i])

    return w, ll_train, ll_test

# 我们训练分类器

alpha = 0.01 # 先试试0.01 XXX 梯度下降的学习率。学生需要完成
n_steps = 100 # epoch训练次数 XXX 基于梯度的优化步数。学生需要完成

X_tilde_train = get_x_tilde(X_train)
X_tilde_test = get_x_tilde(X_test)
w, ll_train, ll_test = fit_w(X_tilde_train, y_train, X_tilde_test, y_test, n_steps, alpha)

##
# 绘制"fit_w"返回的平均对数似然
#
# 输入:
#
# ll: 对数似然值的向量
#
# 输出: 无
#

def plot_ll(ll):
    plt.figure()
    ax = plt.gca()
    plt.xlim(0, len(ll) + 2)
    plt.ylim(min(ll) - 0.1, max(ll) + 0.1)
    ax.plot(np.arange(1, len(ll) + 1), ll, 'r-')
    plt.xlabel('步数')
    plt.ylabel('平均对数似然')
    plt.title('绘制平均对数似然曲线')
    plt.show()

# 我们绘制训练和测试的对数似然值

plot_ll(ll_train)
plot_ll(ll_test)

##
# 绘制逻辑分类器的预测概率
#
# 输入:
#
# X: 数据的输入特征的2维数组（不包括在一开始添加一个常数列的情况）
# y: 数据的类别标签的1维数组（0或1）
# w: 参数向量
# map_inputs: 在原始2D输入上使用基础函数进行扩展的函数。
#
# 输出: 无
#

def plot_predictive_distribution(X, y, w, map_inputs=lambda x: x):
    xx, yy = plot_data_internal(X, y)
    ax = plt.gca()
    X_tilde = get_x_tilde(map_inputs(np.concatenate((xx.ravel().reshape((-1, 1)), yy.ravel().reshape((-1, 1))), 1)))
    Z = predict(X_tilde, w)
    Z = Z.reshape(xx.shape)
    cs2 = ax.contour(xx, yy, Z, cmap='RdBu', linewidths=2)
    plt.clabel(cs2, fmt='%2.1f', colors='k', fontsize=14)
    plt.show()

# 我们绘制预测分布

plot_predictive_distribution(X, y, w)

##
# 用高斯基函数在网格点上评估代替初始输入特征的函数
#
# 输入:
#
# l: 高斯基函数宽度的超参数
# Z: 高斯基函数位置
# X: 评估基函数的点
#
# 输出: 包含高斯基函数评估的特征矩阵。
#

def evaluate_basis_functions(l, X, Z):
    X2 = np.sum(X**2, 1)
    Z2 = np.sum(Z**2, 1)
    ones_Z = np.ones(Z.shape[0])
    ones_X = np.ones(X.shape[0])
    r2 = np.outer(X2, ones_Z) - 2 * np.dot(X, Z.T) + np.outer(ones_X, Z2)
    return np.exp(-0.5 / l**2 * r2)

# 我们扩展数据

l = 0.1 # XXX 高斯基函数的宽度。学生需要完成

X_tilde_train = get_x_tilde(evaluate_basis_functions(l, X_train, X_train))
X_tilde_test = get_x_tilde(evaluate_basis_functions(l, X_test, X_train))

# 我们在基函数扩展的输入上训练新的分类器

alpha = 0.0005 # XXX 基于基函数的梯度下降的学习率。学生需要完成
n_steps = 100 # XXX 基于基函数的梯度下降的步数。学生需要完成

w, ll_train, ll_test = fit_w(X_tilde_train, y_train, X_tilde_test, y_test, n_steps, alpha)

# 我们绘制训练和测试的对数似然值

plot_ll(ll_train)
plot_ll(ll_test)

# 我们绘制预测分布

plot_predictive_distribution(X, y, w, lambda x: evaluate_basis_functions(l, x, X_train))
