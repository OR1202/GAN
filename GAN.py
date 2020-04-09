
from abc import ABCMeta, abstractmethod

import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Input, Dense
from keras.layers.convolutional import Conv2D

from keras.models import clone_model

from keras.models import Model
from keras.optimizers import Adam



class GenerativeAdversarialNetwork(metaclass=ABCMeta):
    """
    GAN
    """
    def __init__(self, input_dim = 100):
        """


        Parameters
        ----------
        input_dim : 潜在空間の次元

        """
        self.input_dim = input_dim
        self.isInit = False

    @abstractmethod
    def Generator(self):
        """
        生成器，詐欺師

        バッチ正規化をすべての層に対して行う。
        活性化関数は出力層はtanh、それ以外はReLuを用いる。
        """
        pass

    def setGenerator(self, generator):
        self.generator = clone_model(generator)

    @abstractmethod
    def Discreminator(self):
        """
        識別器，警察

        バッチ正規化をすべての層に対して行う。
        すべての層の活性関数にLeakyReLuを用いる。
        """
        pass

    def setDiscreminator(self, discreminator):
        self.discreminator = clone_model(discreminator)

    @abstractmethod
    def init(self, X, epochs, batch_size = 32):
        pass


    def fit(self, X, epochs, batch_size = 32, verbose=1, callback=None):
        """
        学習
        ScikitやKerasを模倣

        Parameter
        ---------
        X : ndarray.shape(?, 次元)
        epochs : int
            学習反復回数
        batch_size : int
            データを何個づつ分割するか（1⇒確率的最急降下法，len(X)⇒最急降下法）
            サイズが訓練データのサイズを超えた場合はbatch_size = X.shape[0]
        verbose :
            コンソール
            0 : 学習経過を表示しない
            1 : 誤差を表示
            2 : 誤差とコールバックを表示
        callback : lamda()
            関数（lambda x: print(x)）
        """
        assert X.max() <= 1. and X.min() >= -1., "Xの値を-1~1に正規化してください: (X.astype(np.float32) - 127.5)/127.5"

        # 初回fit時に初期化
        if not self.isInit: self.init(X, epochs, batch_size)
        # 初期化済みの場合は逐次学習

        # 学習開始
        self.train(X, epochs, batch_size = 32, verbose=1, callback = callback)


    def train(self, X, epochs, batch_size = 32, verbose=1, callback=None):
        """
        学習

        Parameter
        ---------
        X : ndarray.shape(?, 特徴ベクトルの形状)
        epochs : int
            学習反復回数
        batch_size : int
            データを何個づつ分割するか（1⇒確率的最急降下法，len(X)⇒最急降下法）
            サイズが訓練データのサイズを超えた場合はbatch_size = X.shape[0]
        verbose :
            コンソール
            0 : 学習経過を表示しない
            1 : 誤差を表示
            2 : 誤差とコールバックを表示
        callback : lamda()
            関数（lambda x: print(x)）
        """
        discreminator_loss = float('inf')
        generator_loss = float('inf')
        for ephoc in range(epochs):

            # 全バッチの平均誤差を表示
            if verbose>0:
                print("学習回数：%5d 偽造判別誤差：%2.10f 偽造生成誤差：%2.10f"% (ephoc+1,np.sum(discreminator_loss), np.sum(generator_loss)))
            if verbose>1 and not callback is None:
                callback(self)

            discreminator_loss = []
            generator_loss = []

            # 訓練データをシャッフル
            Xr = np.random.permutation(X)
            batch = np.array([Xr])
            # バッチに分割
            if batch_size < X.shape[0]:
                batch = np.array_split(Xr, int(X.shape[0] / batch_size), 0)

            # バッチ毎に訓練
            for X_batch in batch:
                d_loss, g_loss = self.train_on_batch(X_batch)
                discreminator_loss.append(d_loss)
                generator_loss.append(g_loss)


    def train_on_batch(self, X):
        """
        バッチ学習
        Kerasを模倣

        Parameter
        ---------
        X : 訓練データ
            X.shape = (?, 特徴ベクトルの形状)
        """

        batch_size = len(X)

        # ノイズをGeneratorに入力し(generator.predict)Generatorの出力を取得
        generator_output = self.generator.predict(self.input(batch_size))

        # データとGeneratorの出力を結合
        X_conc = np.concatenate((X, generator_output))
        Y_conc = [1] * batch_size + [0] * batch_size
        discreminator_loss = self.discreminator.train_on_batch(X_conc, Y_conc)

        # GAN(combined_model, c)に入力してDiscreminatorを学習
        generator_loss = self.dcgan.train_on_batch(self.input(batch_size), [1] * batch_size)

        return discreminator_loss, generator_loss

    def predict(self, X):
        """

        Parameters
        ----------
        X : データ

        Return
        ----------
        データXが機械合成でない確率
        """
        return self.discreminator.predict(X)

    def input(self, batch_size = 1):
        """
        乱数ベクトルの生成

        Parameters
        ----------
        batch_size : ベクトルの件数

        Return
        ----------
        v : ndarray.shape(batch_size, input_dim)（生成画像）
        """
        v = np.random.uniform(-1, 1, size=(batch_size, self.input_dim))
        return v

    def generate(self, X=[]):
        """
        Parameters
        ----------
        X : list（乱数ベクトル）

        Return
        ----------
        生成画像
        """


        if len(X) == 0: # 入力がない場合は乱数を生成
            X = self.input()
        else: # 入力があった場合は検証検証
            assert X.shape[1]==self.input_dim, "ベクトルの長さは%dである必要があります:X.shape[1]=%d"%(self.input_dim, X.shape[1])
            assert X.max() < 1 and X.min() > -1, "Xの値は-1~1となるようにしてください"

        return self.generator.predict(X)


    def save(self, directry = "./", name = "GAN"):
        self.generator.save(directry+name+"_generator.keras")
        self.discreminator.save(directry+name+"_discreminator.keras")

        import pickle
        with open(directry+name+'_model.pickle', 'wb') as f:
            pickle.dump(self, f)


class SimpleGAN(GenerativeAdversarialNetwork):
    """
    ベクトルに対するGAN
    """
    def __init__(self, input_dim = 100):
        self.input_dim = input_dim
        self.isInit = False

    # Override
    def Generator(self):

        input_layer = Input(shape=(self.input_dim,))
        hidden_layer = Dense(
                activation = "relu",
                units = self.input_dim,
                kernel_initializer = "identity"
        )(input_layer)
        hidden_layer = Dense(
                activation = "relu",
                units = self.input_dim,
                kernel_initializer = "identity"
        )(hidden_layer)
        output_layer = Dense(
                activation = "linear",
                units = self.dim,
                kernel_initializer = "identity"
        )(hidden_layer)
        model = Model(inputs = input_layer, outputs = output_layer, name="generator")
        #model.compile(loss='mean_squared_error', optimizer="adam")
        model.summary()

        self.generator = model

    # Override
    def Discreminator(self):
        input_layer = Input(shape=(self.dim,))
        hidden_layer = Dense(
                activation = "relu",
                units = self.input_dim,
                kernel_initializer = "identity"
        )(input_layer)
        hidden_layer = Dense(
                activation = "relu",
                units = self.input_dim,
                kernel_initializer = "identity"
        )(hidden_layer)
        output_layer = Dense(
                activation = "linear",
                units = self.input_dim,
                kernel_initializer = "identity"
        )(hidden_layer)
        model = Model(inputs = input_layer, outputs = output_layer, name="generator")
        #model.compile(loss='mean_squared_error', optimizer="adam")
        model.summary()

        self.discreminator = model


    def init(self, X, epochs, batch_size = 32):

        self.dim = X.shape[1]


        """
        GeneratorとDiscreminator，GANを定義
        """
        self.Generator()
        self.Discreminator()

        from keras.models import Sequential
        self.dcgan = Sequential([self.generator, self.discreminator])

        """
        Generatorをコンパイル
        """
        self.generator.summary()

        """
        GAN構造はDiscreminatorを学習させずにGeneratorにLossを伝搬させるため
        Discriminatorの学習を停止してコンパイル
        """
        self.discreminator.trainable = False
        self.dcgan.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
        self.dcgan.summary()

        """
        Discreminator自体は学習するため
        Discreminatorの学習を可能にしてからコンパイル
        """
        self.discreminator.trainable = True
        self.discreminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
        self.discreminator.summary()


class DCGAN(GenerativeAdversarialNetwork):
    """
    画像に対するCNNを用いたGAN
    """
    def __init__(self, input_dim = 100):
        """

        Parameters
        ----------
        input_dim : 潜在空間の次元

        """
        self.input_dim = input_dim
        self.isInit = False


    def Generator(self):
        """
        生成器，詐欺師

        バッチ正規化をすべての層に対して行う。
        活性化関数は出力層はtanh、それ以外はReLuを用いる。

        notice: ハイパパラメタは基の論文と異なる
        """
        from keras.layers import Reshape, Conv2DTranspose, ReLU, BatchNormalization

        assert self.height % 16 == 0 , "縦横幅は設定した畳み込み層数の二乗で割り切れる値にしてください"
        in_h = int(self.height / 16)
        in_w = int(self.width / 16)

        inputs = Input((self.input_dim,))

        kernel_size = (3,3)
        hidden = Dense(in_h * in_w * 256, name='generator_dense1')(inputs)
        hidden = ReLU()(hidden)
        hidden = Reshape((in_h, in_w, 256), input_shape=(256 * in_h * in_w,))(hidden)
        hidden = Conv2DTranspose(256, kernel_size, strides=2, padding='same', name='generator_conv1')(hidden)
        hidden = BatchNormalization()(hidden)
        hidden = ReLU()(hidden)
        hidden = Conv2DTranspose(128, kernel_size, strides=2, padding='same', name='generator_conv2')(hidden)
        hidden = BatchNormalization()(hidden)
        hidden = ReLU()(hidden)
        hidden = Conv2DTranspose(64, kernel_size, strides=2, padding='same', name='generator_conv3')(hidden)
        hidden = BatchNormalization()(hidden)
        hidden = ReLU()(hidden)

        output = Conv2DTranspose(self.channel, kernel_size=(3, 3), strides=2, padding='same', activation = "tanh", name='generator_out')(hidden)

        model = Model(inputs, output, name='Generator')
        self.generator = model


    def Discreminator(self):
        """
        識別器，警察

        バッチ正規化をすべての層に対して行う。
        すべての層の活性関数にLeakyReLuを用いる。

        notice: ハイパパラメタは基の論文と異なる
        """
        from keras.layers import Flatten, LeakyReLU#, BatchNormalization

        inputs = Input((self.height, self.width, self.channel))

        kernel_size = (3,3)
        hidden = Conv2D(64, kernel_size, strides=2,  padding='same', name='discreminator_conv1')(inputs)
#        hidden = BatchNormalization()(hidden)
        hidden = LeakyReLU(alpha=0.2)(hidden)
        hidden = Conv2D(128, kernel_size, strides=2, padding='same', name='discreminator_conv2')(hidden)
#        hidden = BatchNormalization()(hidden)
        hidden = LeakyReLU(alpha=0.2)(hidden)
        hidden = Conv2D(256, kernel_size, strides=2, padding='same', name='discreminator_conv3')(hidden)
#        hidden = BatchNormalization()(hidden)
        hidden = LeakyReLU(alpha=0.2)(hidden)
        hidden = Flatten()(hidden)

        output = Dense(1, activation='sigmoid', name='discreminator_out')(hidden)

        model = Model(inputs, output, name='Discreminator')

        self.discreminator = model


    def init(self, X, epochs, batch_size = 32):

        self.height = X.shape[1]
        self.width = X.shape[2]
        assert len(X.shape)>3, "チャンネル次元がありません．X = X[:, :, :, None]などを行いX.shape=(PatternNum,height,width,channelNum)となるようにしてください"
        self.channel = X.shape[3]


        """
        GeneratorとDiscreminator，GANを定義
        """
        self.Generator()
        self.Discreminator()

        from keras.models import Sequential
        self.dcgan = Sequential([self.generator, self.discreminator])

        """
        Generatorをコンパイル
        """
        # 事前学習
        self.generator.compile(loss='mean_squared_error', optimizer='adam')
#        self.generator.fit(self.input(X.shape[0]), X, epochs=1000, batch_size = batch_size)
#        for i in range(100):
#            g_error = self.generator.train_on_batch(self.input(batch_size), X[np.random.permutation(len(X))][:batch_size])
#            print("generator",i,g_error)
        self.generator.summary()

        """
        GAN構造はDiscreminatorを学習させずにGeneratorにLossを伝搬させるため
        Discriminatorの学習を停止してコンパイル
        """
        self.discreminator.trainable = False
        self.dcgan.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
        self.dcgan.summary()

        """
        Discreminator自体は学習するため
        Discreminatorの学習を可能にしてからコンパイル
        """
        self.discreminator.trainable = True
        self.discreminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
        # 事前学習
#        self.discreminator.fit(np.concatenate((X, np.random.uniform(-1,1,size=X.shape))), [1]*X.shape[0]+[0]*X.shape[0], epochs=100,batch_size = batch_size)
#        for i in range(100):
#            d_error = self.discreminator.train_on_batch(X[np.random.permutation(len(X))][:batch_size], [1] * batch_size)
#            print("discreminator",i,d_error)
        self.discreminator.summary()

    #Override
    def fit(self, X, epochs, batch_size = 32, verbose=1, callback=None):
        """
        学習
        ScikitやKerasを模倣

        Parameter
        ---------
        X : ndarray.shape(?, 次元)
        epochs : int
            学習反復回数
        batch_size : int
            データを何個づつ分割するか（1⇒確率的最急降下法，len(X)⇒最急降下法）
            サイズが訓練データのサイズを超えた場合はbatch_size = X.shape[0]
        verbose :
            コンソール
            0 : 学習経過を表示しない
            1 : 誤差を表示
            2 : 誤差とコールバックを表示
        callback : lamda()
            関数（lambda x: print(x)）
        """
        assert X.max() <= 1. and X.min() >= -1., "Xの値を-1~1に正規化してください: (X.astype(np.float32) - 127.5)/127.5"

        # 初回fit時に初期化
        if not self.isInit: self.init(X, epochs, batch_size)
        # 初期化済みの場合は逐次学習


        def callback(model):
            if model.channel>1:
                plt.imshow(model.generate()[0][:,:,[2,1,0]])
            else:
                plt.imshow(np.squeeze(model.generate()[0]))
            plt.show()

        # 学習開始
        self.train(X,epochs, batch_size = batch_size, verbose = verbose, callback = callback)

    #Override
    def predict(self, X):
        assert (X.shape[1],X.shape[2]) == (self.height, self.width), "画像の縦横幅が学習時と異なります:(%d, %d)"%(X.shape[1],X.shape[2])
        """

        Parameters
        ----------
        X : 画像

        Return
        ----------
        画像Xが機械合成でない確率
        """
        return self.discreminator.predict(X)

    #Override
    def save(self, directry = "./", name = "DCGAN"):
        self.generator.save(directry+name+"_generator.keras")
        self.discreminator.save(directry+name+"_discreminator.keras")

        import pickle
        with open(directry+name+'_model.pickle', 'wb') as f:
            pickle.dump(self, f)

def load(directry = "./", name = "GAN"):
    import pickle
    with open(directry+name+'_model.pickle', 'rb') as f:
         model = pickle.load(f)

         from keras.models import load_model
         model.generator = load_model(directry+name+"_generator.keras")
         model.discreminator = load_model(directry+name+"_discreminator.keras")

         return model



if __name__ == '__main__':
    """
    cifar100を用いたデモ

    理論：https://www.jstage.jst.go.jp/article/jasj/74/4/74_208/_pdf
    """

    # データ取得
    from keras.datasets import cifar100
    (X_train, Y_train), (X_test, Y_test) = cifar100.load_data()
    del(X_train)
    del(Y_train)
    del(Y_test)

    # 正規化
    X_test = (X_test.astype(np.float32) - 127.5)/127.5 # 値を-1~1の間に設定

    # 処理
    input_dim = 100
    dcgan = DCGAN(input_dim) # 潜在空間の次元を100に設定
    dcgan.fit(X_test, epochs=10, verbose=2)

    # 生成された画像を表示
    G = dcgan.generate(np.random.uniform(low=-1, high=1, size=(10, input_dim)))
    for i, g in enumerate(G):
        plt.gray()
        plt.imshow(1.-np.squeeze(g))
        plt.show()