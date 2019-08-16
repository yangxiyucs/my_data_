from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np


class DCGAN(object):
    """
    Mnist手写数字图片的生成
    """
    def __init__(self):
        # 输入图片的形状
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

    def init_model(self):
        """
        初始化模型的D、G结构
        :return:
        """
        # 定义噪点数据向量长度大小
        self.latent_dim = 100
        # 获取定义好的优化器
        optimizer = Adam(0.0002, 0.5)

        # 1、建立判别结构参数
        self.discriminator = self.build_discriminator()

        self.discriminator.compile(loss="binary_crossentropy",
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # 2、建立生成器结构参数
        # 损失，判别器输出结果与目标值的交叉熵损失
        self.generator = self.build_generator()

        # 定义输出的噪点数据结构，输入到生成器当中，得到图片
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)
        # 图片要输入到判别器得到预测结果
        self.discriminator.trainable = False
        valid = self.discriminator(img)

        # 损失
        self.combined = Model(z, valid)

        self.combined.compile(loss="binary_crossentropy", optimizer=optimizer)

    def build_discriminator(self):
        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(128 * 7 * 7, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((7, 7, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def train(self, epochs, batch_size=32):
        """
        训练D、G结构
        :param epochs: 迭代次数
        :param batch_size: 每次样本数
        :return:
        """
        # 1、加载Mnist数据，处理，目标值建立
        (X_train, _), (_, _) = mnist.load_data()

        # 进行归一化处理
        X_train = X_train / 127.5 - 1.
        print(X_train.shape)

        # 0, 1,2,3 [60000, 28, 28, 1]
        X_train = np.expand_dims(X_train, axis=3)

        # 准备目标值
        # batch_size大小真实样本的目标值1
        # batch_size大小假样本的目标值0
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        # 2、循环迭代训练
        for epoch in range(epochs):

            # 1、训练判别器
            # 准备batch_size个真实样本
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            # 准备batch_size个假样本,[batch_size, ]
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            # 使用generator生成假图片
            gen_imgs = self.generator.predict(noise)

            # 训练
            loss_real = self.discriminator.train_on_batch(imgs, valid)
            loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)

            loss_avg = np.add(loss_real, loss_fake) / 2

            # 2、训练生成器，停止判别器
            # 就是去训练前面指定的conbined模型
            g_loss = self.combined.train_on_batch(noise, valid)

            # 打印结果
            print("迭代次数：%d, 判别器：损失：%f , 生成器：%f" % (epoch, loss_avg[0], g_loss))

            if epoch % 3 == 0:
                self.save_imgs(epoch)

    def save_imgs(self, epoch):

        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("./images/mnist_%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    dc = DCGAN()
    dc.init_model()
    dc.train(epochs=4000, batch_size=32)

