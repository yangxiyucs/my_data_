from keras.layers import Input, Dense, RepeatVector, Concatenate, Dot, Activation
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import LSTM, Bidirectional
from nmt_utils import *


class Seq2seq(object):
    """
    序列模型去进行日期的翻译
    """
    def __init__(self, Tx=30, Ty=10, n_x=32, n_y=64):

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
        获取encoder属性
        :return: None
        """

        self.encoder = Bidirectional(LSTM(self.model_param["n_x"], return_sequences=True, name="bidirectional_1"), merge_mode='concat')

        return None

    def get_decoder(self):
        """
        获取deocder属性
        :return: None
        """

        self.decoder = LSTM(self.model_param["n_y"], return_state=True)

        return None

    def get_output_layer(self):
        """
        获取输出层
        :return: None
        """

        self.output_layer = Dense(self.model_param["y_vocab_size"], activation=softmax)

        return None

    def get_attention(self):
        """
        实现attention的结构属性
        :return: None
        """
        # 1、定义Repeat函数
        repeator = RepeatVector(self.model_param["Tx"])

        # 2、定义Concat函数
        concatenator = Concatenate(axis=-1)

        # 3、定义Dense
        densor1 = Dense(10, activation="tanh", name="Dense1")

        densor2 = Dense(1, activation="relu", name='Dense2')

        # 4、Activatation
        activator = Activation(softmax,
                               name='attention_weights')

        # 5、Dot相当于npt.dot
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

    def init_seq2seq(self):
        """
        初始化网络结构
        :return:
        """
        # 添加encoder属性
        self.get_encoder()

        # 添加decoder属性
        self.get_decoder()

        # 添加attention属性
        self.get_attention()

        # 添加get_output_layer属性
        self.get_output_layer()

        return None

    def computer_one_attention(self, a, s_prev):
        """
        逻辑函数，计算context
        :param a: encoder的所有输出,t'时刻,a=t=1,2,3,4,......Tx
        :param s_prev: decoder的输出，t'-1
        :return: context
        """
        # - 1、扩展s_prev的维度到encoder的所有时刻，编程Tx份
        s_prev = self.attention["repeator"](s_prev)

        # - 2、进行s_prev和a进行拼接
        concat = self.attention["concatenator"]([a, s_prev])

        # - 3、进行全连接计算得到e, 经过激活函数relu计算出e'
        e = self.attention["densor1"](concat)
        en = self.attention["densor2"](e)

        # - 4、e'进过softmax计算，得到系数,每个attention 有Tx个alphas参数
        alphas = self.attention["activator"](en)

        # - 5、系数与a进行计算得到context
        context = self.attention["dotor"]([alphas, a])

        return context

    def model(self):
        """
        定义整个网络模型
        :return: keras 当中的model类型
        """
        # 1、定义encoder的输入X (30, 37)
        X = Input(shape=(self.model_param["Tx"], self.model_param["x_vocab_size"]), name="X")

        # 定义一个初始输入的s0, 64大小
        s0 = Input(shape=(self.model_param["n_y"],), name="s0")
        c0 = Input(shape=(self.model_param["n_y"],), name="c0")
        s = s0
        c = c0

        # 定义一个装有输出的列表
        outputs = []

        # 2、输入到encoder当中，得到a
        a = self.encoder(X)

        # 3、计算输出结果,循环deocder当中t'个时刻，计算每个LSTM的输出结果
        for t in range(self.model_param["Ty"]):

            # (1)循环计算每一个时刻的context
            context = self.computer_one_attention(a, s)

            # （2）输入s0,c0,context到某个时刻decoder得到下次的输出s, c
            # 因为是LSTM结构，所以有两个隐层状态，其中s可以用作输出
            s, _, c = self.decoder(context, initial_state=[s, c])

            # (3)s输出到最后一层softmax得到预测结果
            out = self.output_layer(s)

            outputs.append(out)

        # 输入输出定义好了
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

        output = [dict(zip(self.model_param["y_vocab"].values(), self.model_param["y_vocab"].keys()))[int(i)] for i in
                  prediction]

        print("source:", example)
        print("output:", ''.join(output))

        return None


if __name__ == '__main__':

    s2s = Seq2seq()

    X_onehot, Y_onehot = s2s.load_data(10000)

    s2s.init_seq2seq()

    # s2s.train(X_onehot, Y_onehot)

    s2s.test()


