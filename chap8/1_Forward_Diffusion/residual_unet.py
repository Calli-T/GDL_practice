from sinusoidal_embedding import *

from tensorflow.keras import layers, activations, models


def ResidualBlock(width):
    def apply(x):
        input_width = x.shape[3] # 아마 [batch_size, h, w, c]로 들어오는듯?
        # 아래 if는 채널이 안맞으면 맞춰주는 코드인듯
        if input_width == width:
            residual = x
        else:
            residual = layers.Conv2D(width, kernel_size=1)(x)
        x = layers.BatchNormalization(center=False, scale=False)(x) # 배치정규화
        x = layers.Conv2D(
            width, kernel_size=3, padding="same", activation=activations.swish # ReLU와 비슷하지만 깊은 레이어에 뛰어나다고 하는 Swish 활성화함수?
        )(x)
        x = layers.Conv2D(width, kernel_size=3, padding="same")(x)
        x = layers.Add()([x, residual]) # 잔차연결에서, 잔여물(residual)을 더해주는 과정/이후 스케일 자체는 배치정규화를 하므로 큰 상관이 없는듯?
        return x

    return apply


def DownBlock(width, block_depth):
    def apply(x):
        x, skips = x
        for _ in range(block_depth): # block depth 만큼 residual block을 반복하고
            x = ResidualBlock(width)(x)
            skips.append(x) # upblock에서 concat할 대상인듯, skip은 그냥 입력으로 주는거 씁니다
        x = layers.AveragePooling2D(pool_size=2)(x) # 이후로 크기 2로 풀링, 너비 1/4토막
        return x

    return apply


def UpBlock(width, block_depth):
    def apply(x):
        x, skips = x
        x = layers.UpSampling2D(size=2, interpolation="bilinear")(x) # 다운에서 후풀링 했으니 업에서는 선업스케일링
        for _ in range(block_depth): # 블럭 깊이 만큼 concat-residualblock절차 반복
            x = layers.Concatenate()([x, skips.pop()])
            x = ResidualBlock(width)(x)
        return x

    return apply


# --------------------------------------------------

# U-Net 구축

# 잡음 이미지를 [h, w, 3]으로 만들었다가 합성곱을 거쳐 같은 너비의 32채널 텐서로 만듬
noisy_images = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
x = layers.Conv2D(32, kernel_size=1)(noisy_images)

# 잡음 분산을 32채널로 사인파임베딩,
noise_variances = layers.Input(shape=(1, 1, 1))
noise_embedding = layers.Lambda(sinusoidal_embedding)(noise_variances)
noise_embedding = layers.UpSampling2D(size=IMAGE_SIZE, interpolation="nearest")(
    noise_embedding
)

# downblock에 들어갈 텐서를 concat으로 생성
x = layers.Concatenate()([x, noise_embedding])

skips = []

# 다운블럭 3차례, 그리고 upblock에서 concat할 결과물도 skip에다 챙김
x = DownBlock(32, block_depth=2)([x, skips])
x = DownBlock(64, block_depth=2)([x, skips])
x = DownBlock(96, block_depth=2)([x, skips])

# 잔차 블럭 2회
x = ResidualBlock(128)(x)
x = ResidualBlock(128)(x)

# 업블럭 3차례, 그리고 downblock에서 만든것들 concat함
x = UpBlock(96, block_depth=2)([x, skips])
x = UpBlock(64, block_depth=2)([x, skips])
x = UpBlock(32, block_depth=2)([x, skips])

# 3채널로 압축 땡김
x = layers.Conv2D(3, kernel_size=1, kernel_initializer="zeros")(x)

# 입력이 잡음과 잡음분산, 출력이 x인 모델 생성
unet = models.Model([noisy_images, noise_variances], x, name="unet")

# - 4 -
