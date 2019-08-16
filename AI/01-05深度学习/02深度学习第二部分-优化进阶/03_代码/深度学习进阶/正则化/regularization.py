import numpy as np
import matplotlib.pyplot as plt

from utils import initialize_parameters, compute_cost, forward_propagation, backward_propagation, update_parameters
from utils import load_dataset, predict, sigmoid, relu


# -------
# 1、带有正则化的计算损失函数
# 2、正则化后的反向传播计算
# -------
def compute_cost_with_regularization(A3, Y, parameters, lambd):
    """
    损失函数中增加L2正则化
    """
    m = Y.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]

    # 计算交叉熵损失
    cross_entropy_cost = compute_cost(A3, Y)

    # 开始
    L2_regularization_cost = (1. / m) * (lambd / 2) * \
                             (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3)))

    cost = cross_entropy_cost + L2_regularization_cost
    # 结束

    return cost


def backward_propagation_with_regularization(X, Y, cache, lambd):
    """
    对增加了L2正则化后的损失函数进行反向传播计算
    """

    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache

    dZ3 = A3 - Y

    dW3 = 1. / m * (np.dot(dZ3, A2.T) + lambd * W3)
    db3 = 1. / m * np.sum(dZ3, axis=1, keepdims=True)

    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1. / m * (np.dot(dZ2, A1.T) + lambd * W2)
    db2 = 1. / m * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1. / m * (np.dot(dZ1, X.T) + lambd * W1)
    db1 = 1. / m * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3, "dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1,
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}

    return gradients

# ----------
# 1、droupout前向传播过程
# 2、droupout反向传播过程
# ----------

def forward_propagation_with_dropout(X, parameters, keep_prob=0.5):
    """
    带有dropout的前向传播
    """

    np.random.seed(1)

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    # 计算第一层输出
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)

    # 开始
    # 初始化一个矩阵
    D1 = np.random.rand(A1.shape[0], A1.shape[1])
    # 标记为0和1
    D1 = D1 < keep_prob
    # 对于A1中的部分结果丢弃
    A1 = np.multiply(A1, D1)
    # 保持原来的期望值
    A1 /= keep_prob
    # 结束

    # 计算第二层输出
    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)

    # 开始
    D2 = np.random.rand(A2.shape[0], A2.shape[1])
    D2 = D2 < keep_prob
    A2 = np.multiply(A2, D2)
    A2 /= keep_prob
    # 结束

    # 最后一层输出
    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)

    cache = (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3)

    return A3, cache


def backward_propagation_with_dropout(X, Y, cache, keep_prob):
    """
    droupout的反向传播
    """

    m = X.shape[1]
    (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3) = cache

    dZ3 = A3 - Y
    dW3 = 1. / m * np.dot(dZ3, A2.T)
    db3 = 1. / m * np.sum(dZ3, axis=1, keepdims=True)
    dA2 = np.dot(W3.T, dZ3)
    dA2 = np.multiply(dA2, D2)
    dA2 /= keep_prob

    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1. / m * np.dot(dZ2, A1.T)
    db2 = 1. / m * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dA1 = np.multiply(dA1, D1)
    dA1 /= keep_prob

    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1. / m * np.dot(dZ1, X.T)
    db1 = 1. / m * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3, "dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1,
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}

    return gradients


def model(X, Y, learning_rate=0.3, num_iterations=30000, lambd=0, keep_prob=1):
    """
    使用三层网络，激活函数为：LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID.
    第一个隐层：20个神经元
    第二个隐层：3个神经元
    输出层：1个神经元
    """

    grads = {}
    costs = []
    m = X.shape[1]
    layers_dims = [X.shape[0], 20, 3, 1]

    # 初始化网络参数
    parameters = initialize_parameters(layers_dims)

    # 梯度下降循环逻辑
    for i in range(0, num_iterations):

        # 前向传播计算
        # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID.
        # 如果keep_prob=1，进行正常前向传播
        # 如果keep_prob<1，说明需要进行droupout计算
        if keep_prob == 1:
            a3, cache = forward_propagation(X, parameters)
        elif keep_prob < 1:
            a3, cache = forward_propagation_with_dropout(X, parameters, keep_prob)

        # 计算损失
        # 如果传入lambd不为0，判断加入正则化
        if lambd == 0:
            cost = compute_cost(a3, Y)
        else:
            cost = compute_cost_with_regularization(a3, Y, parameters, lambd)

        # 只允许选择一个，要么L2正则化，要么Droupout
        assert (lambd == 0 or keep_prob == 1)

        if lambd == 0 and keep_prob == 1:
            grads = backward_propagation(X, Y, cache)
        elif lambd != 0:
            grads = backward_propagation_with_regularization(X, Y, cache, lambd)
        elif keep_prob < 1:
            grads = backward_propagation_with_dropout(X, Y, cache, keep_prob)

        # 更新参数
        parameters = update_parameters(parameters, grads, learning_rate)

        # 每10000词打印损失结果
        if i % 10000 == 0:
            print("迭代次数为 {}: 损失结果大小：{}".format(i, cost))
            costs.append(cost)

    # 画出损失变化结果图
    plt.plot(costs)
    plt.ylabel('损失')
    plt.xlabel('迭代次数')
    plt.title("损失变化图，学习率为" + str(learning_rate))
    plt.show()

    return parameters


if __name__ == '__main__':

    train_X, train_Y, test_X, test_Y = load_dataset()

    parameters = model(train_X, train_Y, keep_prob=0.86)

    print("训练集的准确率:")
    predictions_train = predict(train_X, train_Y, parameters)

    print("测试集的准确率:")
    predictions_test = predict(test_X, test_Y, parameters)