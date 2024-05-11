from diff_schedule_func import *

import numpy as np

NOISE_EMBEDDING_SIZE = 32


# 잡음 임베딩 크기 32를 2로 나눈걸 L로 사용
# 0부터 L-1 i에 대해 sin(2π * e^if * x), L부터 2L -1까지는 cos(2π * e^(i-L)f * x)
def sinusoidal_embedding(x):
    frequencies = tf.exp(
        tf.linspace(
            tf.math.log(1.0),
            tf.math.log(1000.0),
            NOISE_EMBEDDING_SIZE // 2,
        )
    )
    angular_speeds = 2.0 * math.pi * frequencies
    embeddings = tf.concat(
        [tf.sin(angular_speeds * x), tf.cos(angular_speeds * x)], axis=3
    )
    return embeddings


# plt로 보여주는 사인파 임베딩을 보여주는 함수
def showSinusoidalEmbedding():
    embedding_list = []
    for y in np.arange(0, 1, 0.01):
        embedding_list.append(sinusoidal_embedding(np.array([[[[y]]]]))[0][0][0])
    embedding_array = np.array(np.transpose(embedding_list))
    fig, ax = plt.subplots()
    ax.set_xticks(
        np.arange(0, 100, 10), labels=np.round(np.arange(0.0, 1.0, 0.1), 1)
    )
    ax.set_ylabel("embedding dimension", fontsize=8)
    ax.set_xlabel("noise variance", fontsize=8)
    plt.pcolor(embedding_array, cmap="coolwarm")
    plt.colorbar(orientation="horizontal", label="embedding value")
    ax.imshow(embedding_array, interpolation="nearest", origin="lower")
    plt.show()

# showSinusoidalEmbedding()

# - 3 -
