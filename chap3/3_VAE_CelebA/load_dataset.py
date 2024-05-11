from tensorflow.keras import utils
import tensorflow as tf
import numpy as np

# parameters
IMAGE_SIZE = 64
BATCH_SIZE = 128

# load data
train_data = utils.image_dataset_from_directory(
    "./img_align_celeba/img_align_celeba",
    labels=None,  # 라벨없음,
    color_mode="rgb",
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42,  # 국룰 히치하이커 시드,
    interpolation="bilinear",
)


# preprocessing
def preprocess(img):
    img = tf.cast(img, "float32") / 255.0
    return img


train = train_data.map(lambda x: preprocess(x))  # 람다식과 사상의 적절한 혼합

# 적절하게 몸비틀어 텐서를 넘파이로 바꾸고 notebooks.util의 sample_batch와 display를 대체
def show_image_from_tensor():
    from matplotlib import pyplot as plt
    plt.figure(figsize=(10, 10))
    for idx, image in enumerate(train_data.as_numpy_iterator()):
        for i in range(100):
            plt.subplot(10, 10, i + 1)
            plt.imshow(image[i].astype(np.uint8))
        break
    plt.show()


# show_image_from_tensor()
# - 1 -