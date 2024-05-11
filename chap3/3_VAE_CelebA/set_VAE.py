from load_dataset import *

from tensorflow.keras import layers, models, losses, metrics
import tensorflow.keras.backend as K

# parameters
CHANNELS = 3
NUM_FEATURES = 64
Z_DIM = 200
BETA = 2000


class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))

        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def get_EnDe():
    # 인코더
    encoder_input = layers.Input(
        shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS), name="encoder_input"
    )
    x = layers.Conv2D(NUM_FEATURES, kernel_size=3, strides=2, padding="same")(
        encoder_input
    )
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(NUM_FEATURES, kernel_size=3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(NUM_FEATURES, kernel_size=3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(NUM_FEATURES, kernel_size=3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(NUM_FEATURES, kernel_size=3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    shape_before_flattening = K.int_shape(x)[1:]  # 디코더에 필요합니다!

    x = layers.Flatten()(x)
    z_mean = layers.Dense(Z_DIM, name="z_mean")(x)
    z_log_var = layers.Dense(Z_DIM, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])

    encoder = models.Model(encoder_input, [z_mean, z_log_var, z], name="encoder")
    # encoder.summary()
    # 가로x세로x채널로 입력받음
    # 합성곱/배치정규화/리키렐루 반복 5회 -> flatten
    # -> z_mean과 z_log_var가 각각 Dense(층이 쌍 꼬리로 갈라짐)
    # -> z_mean/z_log_var와 Sampling함수를 모두 사용해서 잠재공간(의 특징벡터들) 생성

    # 디코더
    decoder_input = layers.Input(shape=(Z_DIM,), name="decoder_input")
    x = layers.Dense(np.prod(shape_before_flattening))(decoder_input)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Reshape(shape_before_flattening)(x)
    x = layers.Conv2DTranspose(
        NUM_FEATURES, kernel_size=3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2DTranspose(
        NUM_FEATURES, kernel_size=3, strides=2, padding="same"
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2DTranspose(
        NUM_FEATURES, kernel_size=3, strides=2, padding="same"
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2DTranspose(
        NUM_FEATURES, kernel_size=3, strides=2, padding="same"
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2DTranspose(
        NUM_FEATURES, kernel_size=3, strides=2, padding="same"
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    decoder_output = layers.Conv2DTranspose(
        CHANNELS, kernel_size=3, strides=1, activation="sigmoid", padding="same"
    )(x)
    decoder = models.Model(decoder_input, decoder_output)

    # decoder.summary()
    # (None/batch_size, dim)의 형식으로 feature vector를 입력으로 받음
    # Dense/배치정규화/리키렐루층에 넣음, 해당 층의 퍼셉트론 수는 flatten 이전의 전체 feature map의 크기 그대로
    # 그걸 다시 flatten 이전의 모양으로 바꿈
    # 전치합성/배치정규화/리키렐루 5회 반복
    # same_padding, 3x3커널과 stride 1인 합성곱, 시그모이드를 통해 결과물 출력

    return encoder, decoder


class VAE(models.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

        # 여기서부터 아래로는 텐서플로우 문법인듯, 잘모르겠음
        # metrics 자체는 척도라는 의미, 아마 손실함수를 정의하는듯
        self.total_loss_tracker = metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def call(self, inputs):
        """특정 입력에서 모델을 호출합니다."""
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        return z_mean, z_log_var, reconstruction

    def train_step(self, data):
        """훈련 스텝을 실행합니다."""
        with tf.GradientTape() as tape:
            z_mean, z_log_var, reconstruction = self(data, training=True)
            reconstruction_loss = tf.reduce_mean(
                BETA * losses.mean_squared_error(data, reconstruction)
            )
            kl_loss = tf.reduce_mean(
                tf.reduce_sum(
                    -0.5
                    * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)),
                    axis=1,
                )
            )
            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        """검증 스텝을 실행합니다."""
        if isinstance(data, tuple):
            data = data[0]

        z_mean, z_log_var, reconstruction = self(data)
        reconstruction_loss = tf.reduce_mean(
            BETA * losses.mean_squared_error(data, reconstruction)
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

    def get_config(self):
        return {}

# - 2 -