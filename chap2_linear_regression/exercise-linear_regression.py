import numpy as np
import matplotlib.pyplot as plt

def load_data(filename):
    """载入数据。"""
    xys = []
    with open(filename, 'r') as f:
        for line in f:
            xys.append(map(float, line.strip().split()))
        xs, ys = zip(*xys)
        return np.asarray(xs), np.asarray(ys)

def identity_basis(x):
    ret = np.expand_dims(x, axis=1)
    return ret

def multinomial_basis(x, feature_num=10):
    '''多项式基函数'''
    x = np.expand_dims(x, axis=1)  # shape(N, 1)
    # 生成多项式特征矩阵，每一列是 x 的不同次幂
    ret = np.hstack([x ** i for i in range(feature_num)])
    return ret

def gaussian_basis(x, feature_num=10):
    '''高斯基函数'''
    x = np.expand_dims(x, axis=1)  # shape(N, 1)
    # 在训练集范围内均匀设置 feature_num 个中心
    centers = np.linspace(0, 25, feature_num)
    sigma = (25 - 0) / feature_num  # 每个高斯的宽度
    # 计算每个样本到每个中心的高斯值
    ret = np.exp(- (x - centers) ** 2 / (2 * sigma ** 2))
    return ret

def main(x_train, y_train):
    """
    训练模型，并返回从x到y的映射。
    """
    basis_func = gaussian_basis  # 可以替换为多项式或高斯基函数
    phi0 = np.expand_dims(np.ones_like(x_train), axis=1)  # 偏置项
    phi1 = basis_func(x_train)  # 特征变换
    phi = np.concatenate([phi0, phi1], axis=1)  # 组合成设计矩阵

    #========== 最小二乘法优化 ==========
    # 公式：w = (phi^T phi)^(-1) phi^T y
    w_ls = np.linalg.inv(phi.T @ phi) @ phi.T @ y_train

    #========== 梯度下降优化 ==========
    w_gd = np.zeros(phi.shape[1])  # 初始化参数
    lr = 1e-1  # 学习率
    epochs = 20000  # 迭代次数
    for i in range(epochs):
        y_pred = phi @ w_gd
        grad = phi.T @ (y_pred - y_train) / len(y_train)
        w_gd -= lr * grad

    # 这里选择返回最小二乘法的结果，也可以返回梯度下降的结果
    w = w_ls  # 或 w = w_gd

    def f(x):
        phi0 = np.expand_dims(np.ones_like(x), axis=1)
        phi1 = basis_func(x)
        phi = np.concatenate([phi0, phi1], axis=1)
        y = np.dot(phi, w)
        return y

    return f

def evaluate(ys, ys_pred):
    """评估模型。"""
    std = np.sqrt(np.mean(np.abs(ys - ys_pred) ** 2))
    return std

# 程序主入口（建议不要改动以下函数的接口）
if __name__ == '__main__':
    train_file = 'train.txt'
    test_file = 'test.txt'
    # 载入数据
    x_train, y_train = load_data(train_file)
    x_test, y_test = load_data(test_file)
    print(x_train.shape)
    print(x_test.shape)

    # 使用线性回归训练模型，返回一个函数f()使得y = f(x)
    f = main(x_train, y_train)

    y_train_pred = f(x_train)
    std = evaluate(y_train, y_train_pred)
    print('训练集预测值与真实值的标准差：{:.1f}'.format(std))
    
    # 计算预测的输出值
    y_test_pred = f(x_test)
    # 使用测试集评估模型
    std = evaluate(y_test, y_test_pred)
    print('预测值与真实值的标准差：{:.1f}'.format(std))

    #显示结果
    plt.plot(x_train, y_train, 'ro', markersize=3)
    # plt.plot(x_test, y_test, 'k')
    plt.plot(x_test, y_test_pred, 'k')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Linear Regression')
    plt.legend(['train', 'test', 'pred'])
    plt.show()