from keras import layers, models, datasets, callbacks, losses, optimizers, metrics

import tensorflow as tf
from tensorflow.keras import backend as K

import numpy as np

from scipy.stats import norm

# - hyperparameter & variables -
IMAGE_SIZE = 32
BATCH_SIZE = 100
VALIDATION_SPLIT = 0.2
EMBEDDING_DIM = 2
EPOCHS = 5
BETA = 500

# - get data & preprocessing -
(x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()


def preprocess(imgs):
    """
    이미지 정규화 및 크기 변경
    """
    imgs = imgs.astype("float32") / 255.0
    imgs = np.pad(imgs, ((0, 0), (2, 2), (2, 2)), constant_values=0.0)
    imgs = np.expand_dims(imgs, -1)
    return imgs


x_train = preprocess(x_train)
x_test = preprocess(x_test)


# - make noise & encoder & decoder -

# 다변량 정규 분포에서 평균 벡터와 분산 행렬을 가지고 샘플링
# z_mean과 z_log_var는 그대로 두고 epsilon만 다변량 정규 분포에서 꺼내와서 분산에 곱함
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        # 추측상으로는 데이터가 [배치 개수, 특징 벡터 차원수]크기인듯?
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))

        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def getEnDe():
    # encoder --------------------------------------------

    encoder_input = layers.Input(
        shape=(32, 32, 1), name='encoder_input'
    )
    x = layers.Conv2D(32, (3, 3), strides=2, activation="relu", padding="same")(encoder_input)
    x = layers.Conv2D(64, (3, 3), strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2D(128, (3, 3), strides=2, activation="relu", padding="same")(x)
    shape_before_flattening = K.int_shape(x)[1:]

    x = layers.Flatten()(x)
    z_mean = layers.Dense(2, name="z_mean")(x)
    z_log_var = layers.Dense(2, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])

    encoder = models.Model(encoder_input, [z_mean, z_log_var, z], name="encoder")
    # print(encoder.summary()

    # decoder --------------------------------------------
    # 디코더는 AE에 쓰던거 그대로 쓰는듯
    decoder_input = layers.Input(shape=(2,), name="decoder_input")
    x = layers.Dense(np.prod(shape_before_flattening))(decoder_input)
    x = layers.Reshape(shape_before_flattening)(x)
    x = layers.Conv2DTranspose(128, (3, 3), strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(64, (3, 3), strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
    decoder_output = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same", name="decoder_output")(x)

    decoder = models.Model(decoder_input, decoder_output)
    # print(decoder.summary())

    return encoder, decoder


# - make VAE -

class VAE(models.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = metrics.Mean(name="kl_loss")

    # @는 파이썬문법, 데코레이터/대상함수를 래핑하고 함수의 앞뒤에 추가적으로 꾸며질 구문을 정의해서 손쉽게 재사용하도록해주는것
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        return z_mean, z_log_var, reconstruction

    # tensorflow-신묘한-기술-학습용
    # reduce_mean은 특정 차원을 제거하고 평균을 구함
    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, reconstruction = self(data)
            reconstruction_loss = tf.reduce_mean(500 * losses.binary_crossentropy(data, reconstruction, axis=(1, 2, 3)))
            kl_loss = tf.reduce_sum(-0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)), axis=1, )
            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {m.name: m.result() for m in self.metrics}

    # 노트북에서 그대로 복붙
    def test_step(self, data):
        """Step run during validation."""
        if isinstance(data, tuple):
            data = data[0]

        z_mean, z_log_var, reconstruction = self(data)
        reconstruction_loss = tf.reduce_mean(
            BETA
            * losses.binary_crossentropy(data, reconstruction, axis=(1, 2, 3))
        )
        kl_loss = tf.reduce_mean(
            tf.reduce_sum(
                -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)),
                axis=1,
            )
        )
        total_loss = reconstruction_loss + kl_loss

        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }

    # 파이썬 문법상 iteration 관련인듯?
    # https://velog.io/@moondeokjong/TF-model.getconfig
    def get_config(self):
        return {}


# - training -
def train():
    en, de = getEnDe()
    vae = VAE(en, de)  # unpacking을 통한 encoder/decoder 넣기
    vae.compile(optimizer="adam")

    # 콜백 둘
    model_checkpoint_callback = callbacks.ModelCheckpoint(
        filepath="./checkpoint",
        save_weights_only=False,
        save_freq="epoch",
        monitor="loss",
        mode="min",
        save_best_only=True,
        verbose=0,
    )  # 모델 체크포인트
    tensorboard_callback = callbacks.TensorBoard(log_dir="./logs")  # 로그작업?
    vae.fit(
        x_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        validation_data=(x_test, x_test),
        callbacks=[model_checkpoint_callback, tensorboard_callback],
    )

    # 최종 모델 저장
    vae.save("./models/vae")
    en.save("./models/encoder")
    de.save("./models/decoder")

    return vae, en, de


# vae, en, de = train()
vae = tf.keras.models.load_model("./models/vae")
en = tf.keras.models.load_model("./models/encoder")
de = tf.keras.models.load_model("./models/decoder")

# - predict -
n_to_predict = 5000
example_images = x_test[:n_to_predict]
example_labels = y_test[:n_to_predict]

z_mean, z_log_var, reconstructions = vae.predict(example_images)  # reconstruction & original image show
print("실제 의류 아이템")
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for idx, img in enumerate(example_images[:100]):
    plt.subplot(10, 10, idx + 1)
    plt.imshow(img, cmap="gray")
plt.show()
print("재구성 이미지")
plt.figure(figsize=(10, 10))
for idx, img in enumerate(reconstructions[:100]):
    plt.subplot(10, 10, idx + 1)
    plt.imshow(img, cmap="gray")
plt.show()

# show latent space
z_mean, z_log_var, feature_vectors = en.predict(example_images)
print(feature_vectors[:10])

figsize = 8
plt.figure(figsize=(figsize, figsize))
plt.scatter(feature_vectors[:, 0], feature_vectors[:, 1], c="black", alpha=0.5, s=3)
plt.show()

# - sampling & decoding to generate -
grid_width, grid_height = (6, 3)
z_sample = np.random.normal(size=(grid_width * grid_height, 2))

reconstructions = de.predict(z_sample)

p = norm.cdf(feature_vectors)
p_sample = norm.cdf(z_sample)

figsize = 8  # 원본과 샘플링된 임베딩 잠재공간에 같이 그려서 보인다
plt.figure(figsize=(figsize, figsize))
plt.scatter(feature_vectors[:, 0], feature_vectors[:, 1], c="black", alpha=0.5, s=3)  # 원본
plt.scatter(z_sample[:, 0], z_sample[:, 1], c="#00B0F0", alpha=1, s=40)  # 생성된 샘플링
plt.show()

fig = plt.figure(figsize=(figsize, grid_height * 2))  # 디코드된 이미지 pyplot으로 보기
fig.subplots_adjust(hspace=0.4, wspace=0.4)

for i in range(grid_width * grid_height):
    ax = fig.add_subplot(grid_height, grid_width, i + 1)
    ax.axis("off")
    ax.text(
        0.5,
        -0.35,
        str(np.round(z_sample[i, :], 1)),
        fontsize=10,
        ha="center",
        transform=ax.transAxes,
    )
    ax.imshow(reconstructions[i, :, :], cmap="Greys")

plt.show()

# - assign colors to fashion classes in latent space and visualize -
# 레이블(의류 종류)에 따라 임베딩에 색상을 지정합니다.
figsize = 8
fig = plt.figure(figsize=(figsize * 2, figsize))
ax = fig.add_subplot(1, 2, 1)
plot_1 = ax.scatter(
    feature_vectors[:, 0], feature_vectors[:, 1], cmap="rainbow", c=example_labels, alpha=0.8, s=3
)
plt.colorbar(plot_1)
ax = fig.add_subplot(1, 2, 2)
plot_2 = ax.scatter(
    p[:, 0], p[:, 1], cmap="rainbow", c=example_labels, alpha=0.8, s=3
)
plt.show()

# - 정규화된 클래스별 잠재공간에서 색상표 -
figsize = 12
grid_size = 15
plt.figure(figsize=(figsize, figsize))
plt.scatter(
    p[:, 0], p[:, 1], cmap="rainbow", c=example_labels, alpha=0.8, s=300
)
plt.colorbar()

x = norm.ppf(np.linspace(0, 1, grid_size))
y = norm.ppf(np.linspace(1, 0, grid_size))
xv, yv = np.meshgrid(x, y)
xv = xv.flatten()
yv = yv.flatten()
grid = np.array(list(zip(xv, yv)))

reconstructions = de.predict(grid) # plt.scatter(grid[:, 0], grid[:, 1], c="black", alpha=1, s=10)
plt.show()

# - 특징벡터 좌표(x, y)별 디코드 이미지 -
fig = plt.figure(figsize=(figsize, figsize))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i in range(grid_size**2):
    ax = fig.add_subplot(grid_size, grid_size, i + 1)
    ax.axis("off")
    ax.imshow(reconstructions[i, :, :], cmap="Greys")
plt.show()