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
        """
        初始化自动编码器模型
        将编码器和解码器放在一起作为一个模型
        :return: auto_encoder
        """
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
        (x_train, _), (x_test, _) = mnist.load_data()

        # 进行归一化
        x_train = x_train.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.

        # 进行形状改变
        x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
        x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
        print(x_train.shape)
        print(x_test.shape)

        # 添加噪音
        x_train_noisy = x_train + np.random.normal(loc=0.0, scale=3.0, size=x_train.shape)
        x_test_noisy = x_test + np.random.normal(loc=0.0, scale=3.0, size=x_test.shape)

        # 重新进行限制每个像素值的大小在0~1之间
        x_train_noisy = np.clip(x_train_noisy, 0., 1.)
        x_test_noisy = np.clip(x_test_noisy, 0., 1.)

        # 训练
        self.model.fit(x_train, x_train,
                       epochs=2,
                       batch_size=256,
                       shuffle=True,
                       validation_data=(x_test, x_test))

    def display(self):
        """
        显示前后效果对比
        :return:
        """
        (x_train, _), (x_test, _) = mnist.load_data()

        x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

        x_test_noisy = x_test + np.random.normal(loc=3.0, scale=10.0, size=x_test.shape)

        decoded_imgs = self.model.predict(x_test_noisy)

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
    # ae.train()
    ae.display()



