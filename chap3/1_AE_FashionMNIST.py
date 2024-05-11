import numpy as np
from tensorflow.keras import datasets
from tensorflow.keras import layers
from keras import backend as K
from tensorflow.keras.models import Model
from matplotlib import pyplot as plt
import cv2


def preprocess(imgs):
    imgs = imgs.astype("float32") / 255.00
    imgs = np.pad(imgs, ((0, 0), (2, 2), (2, 2)), constant_values=0.0)
    imgs = np.expand_dims(imgs, -1)

    return imgs


'''def get_encoder_with_shape():
    encoder_input = layers.Input(shape=(32, 32, 1), name="encoder_input")
    x = layers.Conv2D(32, (3, 3), strides=2, activation="relu", padding="same")(encoder_input)
    x = layers.Conv2D(64, (3, 3), strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2D(128, (3, 3), strides=2, activation="relu", padding="same")(x)
    shape_before_flatting = K.int_shape(x)[1:]

    x = layers.Flatten()(x)
    encoder_output = layers.Dense(2, name="encoder_output")(x)

    encoder = Model(encoder_input, encoder_output)

    return encoder, shape_before_flatting


def get_decoder(shape_before_flatting):
    decoder_input = layers.Input(shape=(2,), name="decoder_input")
    x = layers.Dense(np.prod(shape_before_flatting))(decoder_input)
    x = layers.Reshape(shape_before_flatting)(x)
    x = layers.Conv2DTranspose(128, (3, 3), strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(64, (3, 3), strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
    decoder_output = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same", name="decoder_output")(x)

    decoder = Model(decoder_input, decoder_output)

    return decoder'''


# en-de 연결된 모델
def getAutoEncoder():
    encoder_input = layers.Input(shape=(32, 32, 1), name="encoder_input")
    x = layers.Conv2D(32, (3, 3), strides=2, activation="relu", padding="same")(encoder_input)
    x = layers.Conv2D(64, (3, 3), strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2D(128, (3, 3), strides=2, activation="relu", padding="same")(x)
    shape_before_flatting = K.int_shape(x)[1:]

    x = layers.Flatten()(x)
    encoder_output = layers.Dense(2, name="encoder_output")(x)
    encoder = Model(encoder_input, encoder_output)

    decoder_input = layers.Input(shape=(2,), name="decoder_input")
    x = layers.Dense(np.prod(shape_before_flatting))(decoder_input)
    x = layers.Reshape(shape_before_flatting)(x)
    x = layers.Conv2DTranspose(128, (3, 3), strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(64, (3, 3), strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
    decoder_output = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same", name="decoder_output")(x)
    decoder = Model(decoder_input, decoder_output)

    return Model(encoder_input, decoder(encoder_output)), encoder, decoder


# get images & preprocess images
(x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()

x_train = preprocess(x_train)
x_test = preprocess(x_test)

# get AE & compile
autoencoder, encoder, decoder = getAutoEncoder()
# binary_cross는 0.5에 가까운 예측이 낮은손실을 만들기때문에, rmse보다 조금 흐릿한편 대신 픽셀격자가 선명
autoencoder.compile(optimizer="adam", loss="binary_crossentropy")

# train
autoencoder.fit(x_train, x_train, epochs=5, batch_size=100, shuffle=True,
                validation_data=(x_test[:5000], x_test[:5000]))
# 원본 코드에서 test로 나눠둔 자료를 val로 쓰는듯?, 테스트용 데이터는 좀 잘라서 따로 코드 써놓음
# 두 번 쓴건 입력과 정답이 같다는 뜻/당연히 입력과 비슷하게 재구성해야함

# 이미지 재구성과 출력(AE 모델에 원본이미지를 넣는것)
example_images = x_test[5000:]  # 0~4999 val에 썼으니 5000~9999번은 생성에 써보자
predictions = autoencoder.predict(example_images)

plt.figure(figsize=(10, 10))
for idx, img in enumerate(example_images[:100]):
    plt.subplot(10, 10, idx + 1)
    plt.imshow(img, cmap="gray")

plt.figure(figsize=(10, 10))
for idx, img in enumerate(predictions[:100]):
    plt.subplot(10, 10, idx + 1)
    plt.imshow(img, cmap="gray")

plt.show()

# 잠재공간 시각화
# example_images = x_test[:9900]
embeddings = encoder.predict(example_images)  # 실제 val에 쓰인거 잠재공간에서 보자
example_labels = y_test[:5000]
plt.figure(figsize=(8, 8))
plt.scatter(embeddings[:, 0], embeddings[:, 1], c=example_labels, alpha=0.8, s=3)
plt.show()

## 새로운 이미지 생성하기
# 기존 임베딩 범위 상한 하한
mins, maxs = np.min(embeddings, axis=0), np.max(embeddings, axis=0)
print(mins, maxs)
# 범위 안에서 무작위 샘플링
sample = np.random.uniform(mins, maxs, size=(18, 2))  # size=(grid_width * grid_height, EMBEDDING_DIM)

# 샘플링한거 디코더에 넣기
reconstructions = decoder.predict(sample)

## 산점도내에서 임베딩 기존과 생성한거 동시에 표기
plt.figure(figsize=(8, 8))
# 기존 임베딩은 흑색으로 처리, 새로만든건 컬러로 병기
plt.scatter(embeddings[:, 0], embeddings[:, 1], c="black", alpha=0.5, s=2)
plt.scatter(sample[:, 0], sample[:, 1], c="#00B0F0", alpha=1, s=40)
plt.show()

## 디코딩을 거쳐 새로 생성된 이미지를 출력
fig = plt.figure(figsize=(8, 3 * 2))
fig.subplots_adjust(hspace=0.4, wspace=0.4)

for i in range(6 * 3):
    ax = fig.add_subplot(6, 3, i + 1)
    ax.axis("off")
    ax.text(
        0.5,
        -0.35,
        str(np.round(sample[i, :], 1)),
        fontsize=10,
        ha="center",
        transform=ax.transAxes,
    )
    ax.imshow(reconstructions[i, :, :], cmap="Greys")

plt.show()
