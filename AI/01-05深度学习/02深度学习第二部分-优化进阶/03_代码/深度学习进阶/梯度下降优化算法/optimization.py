import numpy as np
import matplotlib.pyplot as plt
import math
import sklearn
import sklearn.datasets

from utils import initialize_parameters, forward_propagation, compute_cost, backward_propagation
from utils import load_dataset, predict


def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    """
    创建每批次固定数量特征值和目标值
    """

    np.random.seed(seed)
    m = X.shape[1]
    mini_batches = []

    # 对所有数据进行打乱
    permutation = list(np.random.permutation(m))

    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1, m))

    # 循环将每批次数据按照固定格式装进列表当中
    num_complete_minibatches = math.floor(
        m / mini_batch_size)

    # 所有训练数据分成多少组
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size: (k + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size: (k + 1) * mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # 最后剩下的样本数量mini-batch < mini_batch_size
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size:]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def initialize_momentum(parameters):
    """
    初始化网络中每一层的动量梯度下降的指数加权平均结果参数
    parameters['W' + str(l)] = Wl
    parameters['b' + str(l)] = bl
    return:
    v['dW' + str(l)] = velocity of dWl
    v['db' + str(l)] = velocity of dbl
    """
    # 得到网络的层数
    L = len(parameters) // 2
    v = {}

    # 初始化动量参数
    for l in range(L):
        v["dW" + str(l + 1)] = np.zeros(parameters['W' + str(l + 1)].shape)
        v["db" + str(l + 1)] = np.zeros(parameters['b' + str(l + 1)].shape)

    return v


def update_parameters_with_momentum(parameters, gradients, v, beta, learning_rate):
    """
    动量梯度下降算法实现
    """
    # 得到网络的层数
    L = len(parameters) // 2

    # 动量梯度参数更新
    for l in range(L):

        # 开始
        v["dW" + str(l + 1)] = beta * v['dW' + str(l + 1)] + (1 - beta) * (gradients['dW' + str(l + 1)])
        v["db" + str(l + 1)] = beta * v['db' + str(l + 1)] + (1 - beta) * (gradients['db' + str(l + 1)])
        parameters["W" + str(l + 1)] = parameters['W' + str(l + 1)] - learning_rate * v['dW' + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters['b' + str(l + 1)] - learning_rate * v['db' + str(l + 1)]
        # 结束

    return parameters, v


def initialize_adam(parameters):
    """
    初始化Adam算法中的参数
    """
    # 得到网络的参数
    L = len(parameters) // 2
    v = {}
    s = {}

    # 利用输入，初始化参数v,s
    for l in range(L):

        v["dW" + str(l + 1)] = np.zeros(parameters['W' + str(l + 1)].shape)
        v["db" + str(l + 1)] = np.zeros(parameters['b' + str(l + 1)].shape)
        s["dW" + str(l + 1)] = np.zeros(parameters['W' + str(l + 1)].shape)
        s["db" + str(l + 1)] = np.zeros(parameters['b' + str(l + 1)].shape)

    return v, s


def update_parameters_with_adam(parameters, gradients, v, s, t, learning_rate=0.01,
                                beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    更新Adam算法网络的参数
    """
    # 网络大小
    L = len(parameters) // 2
    v_corrected = {}
    s_corrected = {}

    # 更新所有参数
    for l in range(L):
        # 对梯度进行移动平均计算. 输入: "v, gradients, beta1". 输出: "v".
        # 开始
        v["dW" + str(l + 1)] = beta1 * v['dW' + str(l + 1)] + (1 - beta1) * gradients['dW' + str(l + 1)]
        v["db" + str(l + 1)] = beta1 * v['db' + str(l + 1)] + (1 - beta1) * gradients['db' + str(l + 1)]
        # 结束

        # 计算修正结果. 输入: "v, beta1, t". 输出: "v_corrected".
        # 开始
        v_corrected["dW" + str(l + 1)] = v['dW' + str(l + 1)] / (1 - np.power(beta1, t))
        v_corrected["db" + str(l + 1)] = v['db' + str(l + 1)] / (1 - np.power(beta1, t))
        # 结束

        # 平方梯度的移动平均值. 输入: "s, gradients, beta2". 输出: "s".
        # 开始
        s["dW" + str(l + 1)] = beta2 * s['dW' + str(l + 1)] + (1 - beta2) * np.power(gradients['dW' + str(l + 1)], 2)
        s["db" + str(l + 1)] = beta2 * s['db' + str(l + 1)] + (1 - beta2) * np.power(gradients['db' + str(l + 1)], 2)
        # 结束

        # 计算修正的结果. 输入: "s, beta2, t". 输出: "s_corrected".
        # 开始
        s_corrected["dW" + str(l + 1)] = s['dW' + str(l + 1)] / (1 - np.power(beta2, t))
        s_corrected["db" + str(l + 1)] = s['db' + str(l + 1)] / (1 - np.power(beta2, t))
        # 结束

        # 更新参数. 输入: "parameters, learning_rate, v_corrected, s_corrected, epsilon". 输出: "parameters".
        # 开始
        parameters["W" + str(l + 1)] = parameters['W' + str(l + 1)] - learning_rate * v_corrected[
            'dW' + str(l + 1)] / np.sqrt(s_corrected['dW' + str(l + 1)] + epsilon)
        parameters["b" + str(l + 1)] = parameters['b' + str(l + 1)] - learning_rate * v_corrected[
            'db' + str(l + 1)] / np.sqrt(s_corrected['db' + str(l + 1)] + epsilon)
        # 结束

    return parameters, v, s


def model(X, Y, optimizer, learning_rate=0.0007, mini_batch_size=64, beta=0.9,
          beta1=0.9, beta2=0.999, epsilon=1e-8, num_epochs=10000, print_cost=True):
    """
    模型逻辑
    定义一个三层网络（不包括输入层）
    第一个隐层：5个神经元
    第二个隐层：2个神经元
    输出层：1个神经元
    """
    # 计算网络的层数
    layers_dims = [train_X.shape[0], 5, 2, 1]

    L = len(layers_dims)
    costs = []
    t = 0
    seed = 10

    # 初始化网络结构
    parameters = initialize_parameters(layers_dims)

    # 初始化优化器参数
    if optimizer == "momentum":
        v = initialize_momentum(parameters)
    elif optimizer == "adam":
        v, s = initialize_adam(parameters)

    # 优化逻辑
    for i in range(num_epochs):

        # 每次迭代所有样本顺序打乱不一样
        seed = seed + 1
        # 获取每批次数据
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)

        # 开始

        for minibatch in minibatches:

            # Mini-batch每批次的数据
            (minibatch_X, minibatch_Y) = minibatch

            # 前向传播minibatch_X, parameters，返回a3, caches
            a3, caches = forward_propagation(minibatch_X, parameters)

            # 计算损失，a3, minibatch_Y，返回cost
            cost = compute_cost(a3, minibatch_Y)

            # 反向传播，返回梯度
            gradients = backward_propagation(minibatch_X, minibatch_Y, caches)

            # 更新参数
            if optimizer == "momentum":

                parameters, v = update_parameters_with_momentum(parameters, gradients, v, beta, learning_rate)

            elif optimizer == "adam":
                t = t + 1
                parameters, v, s = update_parameters_with_adam(parameters, gradients, v, s,
                                                               t, learning_rate, beta1, beta2, epsilon)

        # 结束

        # 每个1000批次打印损失
        if print_cost and i % 1000 == 0:
            print("第 %i 次迭代的损失值: %f" % (i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    # 画出损失的变化
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('epochs (per 100)')
    plt.title("损失图")
    plt.show()

    return parameters


if __name__ == '__main__':

    train_X, train_Y = load_dataset()

    parameters = model(train_X, train_Y, optimizer="momentum")

    predictions = predict(train_X, train_Y, parameters)
