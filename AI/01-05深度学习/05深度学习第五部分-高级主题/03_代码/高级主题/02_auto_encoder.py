from keras.layers import Input, Dense
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt


class AutoEncoder(object):
    """自动编码器
    """
    def __init__(self):

        self.encoding_dim = 32
        self.decoding_dim = 784

        self.model = self.auto_encoder_model()

    def auto_encoder_model(self):
        """自编码器的结构
        """
        # # 编码器输入结构
        # input_img = Input(shape=(784, ))
        #
        # # 定义编码器：输出32个神经元，使用relu激活函数，（32
        # # 这个值可以自己制定）
        # encoder = Dense(self.encoding_dim, activation='relu')(input_img)
        #
        # # 定义解码器：输出784个神经元，使用sigmoid函数，（784
        # # 这个值是输出与原图片大小一致）
        # decoder = Dense(self.decoding_dim, activation='sigmoid')(encoder)
        #
        # # 定义完整的模型逻辑
        # auto_encoder = Model(inputs=input_img, outputs=decoder)
        #
        # auto_encoder.compile(optimizer='adam', loss="binary_crossentropy")

        # 2、深度自编码器
        # input_img = Input(shape=(784,))
        # encoded = Dense(128, activation='relu')(input_img)
        # encoded = Dense(64, activation='relu')(encoded)
        # encoded = Dense(32, activation='relu')(encoded)
        #
        # decoded = Dense(64, activation='relu')(encoded)
        # decoded = Dense(128, activation='relu')(decoded)
        # decoded = Dense(784, activation='sigmoid')(decoded)
        #
        # auto_encoder = Model(input=input_img, output=decoded)
        # auto_encoder.compile(optimizer='adam', loss='binary_crossentropy')

        # 3、卷积自动编码器
        input_img = Input(shape=(28, 28, 1))

        x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)
        print(encoded)

        x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
        print(decoded)

        auto_encoder = Model(input_img, decoded)
        auto_encoder.compile(optimizer='adam', loss='binary_crossentropy')

        return auto_encoder

    def train(self):
        """
        训练自编码器
        :return:
        """
        # - 读取Mnist数据，并进行归一化处理以及形状修改
        (x_train, _), (x_test, _) = mnist.load_data()

        x_train = x_train.astype("float32") / 255.
        x_test = x_test.astype("float32") / 255.

        # 1、由于全连接层的要求，需要将数据装换成二维的[batch, feature]
        # [60000, 784]
        # [10000, 784]
        # x_train = np.reshape(x_train, (len(x_train), np.prod(x_train.shape[1:])))
        # x_test = np.reshape(x_test, (len(x_test), np.prod(x_test.shape[1:])))
        # print(x_train.shape)
        # print(x_test.shape)

        # 2、卷积网络处理， 由于卷积层的要求
        # [60000, 28, 28, 1]
        # [10000, 28, 28, 1]
        x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
        x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
        print(x_train.shape)
        print(x_test.shape)

        # 进行噪点数据处理
        x_train_noisy = x_train + np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
        x_test_noisy = x_test + np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

        # 处理成0~1之间的数据
        x_train_noisy = np.clip(x_train_noisy, 0., 1.)
        x_test_noisy = np.clip(x_test_noisy, 0., 1.)

        # - 模型进行fit训练
        # - 指定迭代次数
        # - 指定每批次数据大小
        # - 是否打乱数据
        # - 验证集合
        self.model.fit(x_train_noisy, x_train,
                       epochs=5,
                       batch_size=256,
                       shuffle=True,
                       validation_data=(x_test_noisy, x_test))

    def display(self):
        """
        显示前后效果对比
        :return:
        """
        (x_train, _), (x_test, _) = mnist.load_data()

        # 普通自编码器
        # x_test = np.reshape(x_test, (len(x_test), np.prod(x_test.shape[1:])))

        # 卷积自编码器
        x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

        # 处理噪点数据
        x_test_noisy = x_test + np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

        decoded_imgs = self.model.predict(x_test)

        plt.figure(figsize=(20, 4))
        # 显示5张结果
        n = 5
        for i in range(n):
            # 显示编码前结果
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(x_test_noisy[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # 显示编解码后结果
            ax = plt.subplot(2, n, i + n + 1)
            plt.imshow(decoded_imgs[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()


if __name__ == '__main__':
    ae = AutoEncoder()
    ae.train()
    ae.display()