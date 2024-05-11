from tensorflow.keras import utils
import tensorflow as tf
from colab_utils import sample_batch, display

IMAGE_SIZE = 64
BATCH_SIZE = 64
DATASET_REPETITIONS = 5

# 데이터 로드
train_data = utils.image_dataset_from_directory(
    "./datasets/",
    labels=None,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=None,
    shuffle=True,
    seed=42,
    interpolation="bilinear",
)


# 데이터 전처리
def preprocess(img):
    img = tf.cast(img, "float32") / 255.0
    return img


# Scaling / Repetition / Batch Processing
train = train_data.map(lambda x: preprocess(x))
train = train.repeat(DATASET_REPETITIONS)
train = train.batch(BATCH_SIZE, drop_remainder=True)

'''
# 훈련 데이터셋의 꽃 이미지 일부를 출력
train_sample = sample_batch(train)
display(train_sample)
'''

# - 1 -
