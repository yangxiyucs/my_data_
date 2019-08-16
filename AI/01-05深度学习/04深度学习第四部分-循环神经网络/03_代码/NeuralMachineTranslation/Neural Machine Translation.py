from keras.layers import Bidirectional, Concatenate, Dot, Input, LSTM
from keras.layers import RepeatVector, Dense, Activation
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model, Model
from nmt_utils import *
import numpy as np


class Seq2seq(object):
    """Seq2seq进行日期格式翻译
    """
    def __init__(self, Tx=30, Ty=10, n_x=32, n_y=64):
        # 定义网络的相关参数
        self.model_param = {
            "Tx": Tx,  # 定义encoder序列最大长度
            "Ty": Ty,  # decoder序列最大长度
            "n_x": n_x,  # encoder的隐层输出值大小
            "n_y": n_y  # decoder的隐层输出值大小和cell输出值大小
        }

    def load_data(self, m):
        """
        指定获取m条数据
        :param m: 数据的总样本数
        :return:
            dataset:[('9 may 1998', '1998-05-09'), ('10.09.70', '1970-09-10')]
            x_vocab:翻译前的格式对应数字{' ': 0, '.': 1, '/': 2, '0': 3, '1': 4, '2': 5, '3': 6, '4': 7,....}
            y_vocab:翻译后的格式对应数字{'-': 0, '0': 1, '1': 2, '2': 3, '3': 4, '4': 5, '5': 6, '6': 7, '7': 8, '8': 9, '9': 10}
        """
        # 获取3个值：数据集，特征词的字典映射，目标词字典映射
        dataset, x_vocab, y_vocab = load_dataset(m)

        # 获取处理好的数据：特征x以及目标y的one_hot编码
        X, Y, X_onehot, Y_onehot = preprocess_data(dataset, x_vocab, y_vocab, self.model_param["Tx"], self.model_param["Ty"])

        print("整个数据集特征值的形状:", X_onehot.shape)
        print("整个数据集目标值的形状:", Y_onehot.shape)

        # 打印数据集
        print("查看第一条数据集格式：特征值:%s, 目标值: %s" % (dataset[0][0], dataset[0][1]))
        print(X[0], Y[0])
        print("one_hot编码：", X_onehot[0], Y_onehot[0])

        # 添加特征词个不重复个数以及目标词的不重复个数
        self.model_param["x_vocab"] = x_vocab
        self.model_param["y_vocab"] = y_vocab

        self.model_param["x_vocab_size"] = len(x_vocab)
        self.model_param["y_vocab_size"] = len(y_vocab)

        return X_onehot, Y_onehot

    def get_encoder(self):
        """
        定义编码器结构
        :return:
        """
        # 指定隐层值输出的大小

        self.encoder = Bidirectional(LSTM(self.model_param["n_x"], return_sequences=True, name='bidirectional_1'),
                                     merge_mode='concat')

        return None

    def get_decoder(self):
        """
        定义解码器结构
        :return:
        """
        # 定义decoder结构，指定隐层值的形状大小，return_state=True
        self.decoder = LSTM(self.model_param["n_y"], return_state=True)

        return None

    def get_attention(self):
        """
        定义Attention的结构
        :return: attention结构
        """

        repeator = RepeatVector(self.model_param["Tx"])

        concatenator = Concatenate(axis=-1)

        densor1 = Dense(10, activation="tanh", name='Dense1')

        densor2 = Dense(1, activation="relu", name='Dense2')

        activator = Activation(softmax,
                               name='attention_weights')
        dotor = Dot(axes=1)

        # 将结构存储在attention当中
        self.attention = {
            "repeator": repeator,
            "concatenator": concatenator,
            "densor1": densor1,
            "densor2": densor2,
            "activator": activator,
            "dotor": dotor
        }

        return None

    def get_output_layer(self):
        """
        定义输出层
        :return: output_layer
        """

        # 对decoder输出进行softmax，输出向量大小为y_vocab大小
        self.output_layer = Dense(self.model_param["y_vocab_size"], activation=softmax)

        return None

    def init_seq2seq(self):
        """
        初始化网络结构
        :return:
        """
        self.get_encoder()

        self.get_decoder()

        self.get_attention()

        self.get_output_layer()

        return None

    def computer_one_attention(self, a, s_prev):
        """
        利用定义好的attention结构计算中的alpha系数与a对应输出
        :param a:隐层状态值 (m, Tx, 2*n_a)
        :param s_prev: LSTM的初始隐层状态值， 形状(sample, n_s)
        :return: context
        """
        # 使用repeator扩大数据s_prev的维度为(sample, Tx, n_y)，这样可以与a进行合并
        s_prev = self.attention["repeator"](s_prev)

        # 将a和s_prev 按照最后一个维度进行合并计算
        concat = self.attention["concatenator"]([a, s_prev])

        # 使用densor1全连接层网络计算出e
        e = self.attention["densor1"](concat)

        # 使用densor2增加relu激活函数计算
        energies = self.attention["densor2"](e)

        # 使用"activator"的softmax函数计算权重"alphas"
        # 这样一个attention的系数计算完成
        alphas = self.attention["activator"](energies)

        # 使用dotor,矩阵乘法，将 "alphas" and "a" 去计算context/c
        context = self.attention["dotor"]([alphas, a])

        return context

    def model(self):
        """
        定义模型获取模型实例
        :param model_param: 网络的相关参数
        :param seq2seq:网络结构
        :return: model,Keras model instance
        """
        # 定义模型的输入 (Tx,)
        # 定义decoder中隐层初始状态值s0以及cell输出c0
        X = Input(shape=(self.model_param["Tx"], self.model_param["x_vocab_size"]), name='X')

        s0 = Input(shape=(self.model_param["n_y"],), name='s0')
        c0 = Input(shape=(self.model_param["n_y"],), name='c0')
        s = s0
        c = c0

        # 定义装有输出值的列表
        outputs = []

        # 步骤1：定义encoder的双向LSTM结构得输出a
        a = self.encoder(X)

        # 步骤3：循环decoder的Ty次序列输入，获取decoder最后输出
        # 包括计算Attention输出
        for t in range(self.model_param["Ty"]):
            # 1: 定义decoder第t'时刻的注意力结构并输出context
            context = self.computer_one_attention(a, s)

            # 2: 对"context" vector输入到deocder当中
            # 获取cell的两个输出隐层状态和，initial_state= [previous hidden state, previous cell state]
            s, _, c = self.decoder(context, initial_state=[s, c])

            # 3: 应用 Dense layere获取deocder的t'时刻的输出
            out = self.output_layer(s)

            # 4: 将decoder中t'时刻的输出装入列表
            outputs.append(out)

        # 步骤 4: 创建model实例，定义输入输出
        model = Model(inputs=(X, s0, c0), outputs=outputs)

        return model

    def train(self, X_onehot, Y_onehot):
        """
        训练
        :param X_onehot: 特征值的one_hot编码
        :param Y_onehot: 目标值的one_hot编码
        :return:
        """
        # 利用网络结构定义好模型输入输出
        model = self.model()

        opt = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.001)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        s0 = np.zeros((10000, self.model_param["n_y"]))
        c0 = np.zeros((10000, self.model_param["n_y"]))
        outputs = list(Y_onehot.swapaxes(0, 1))

        # 输入x,以及decoder中LSTM的两个初始化值
        model.fit([X_onehot, s0, c0], outputs, epochs=1, batch_size=100)

        return None

    def test(self):
        """
        测试
        :return:
        """
        model = self.model()

        model.load_weights("./models/model.h5")

        example = '1 March 2001'
        source = string_to_int(example, self.model_param["Tx"], self.model_param["x_vocab"])
        source = np.expand_dims(np.array(list(map(lambda x:
                                                  to_categorical(x, num_classes=self.model_param["x_vocab_size"]),
                                                  source))), axis=0)
        s0 = np.zeros((10000, self.model_param["n_y"]))
        c0 = np.zeros((10000, self.model_param["n_y"]))
        prediction = model.predict([source, s0, c0])
        prediction = np.argmax(prediction, axis=-1)

        output = [dict(zip(self.model_param["y_vocab"].values(), self.model_param["y_vocab"].keys()))[int(i)] for i in prediction]

        print("source:", example)
        print("output:", ''.join(output))

        return None


if __name__ == '__main__':

    s2s = Seq2seq()

    X_onehot, Y_onehot = s2s.load_data(10000)

    # s2s.init_seq2seq()

    # s2s.train(X_onehot, Y_onehot)

    # s2s.test()

