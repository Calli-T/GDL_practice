from train import *
from utils import *

import pandas as pd

# 모델 가져오기
vae = tf.keras.models.load_model("./models/vae")
en = tf.keras.models.load_model("./models/encoder")
de = tf.keras.models.load_model("./models/decoder")

# - 4.2 -

# 열은 특징 행은 사진, 셀럽 사진에서 외형적 특징을 뭐뭐 가지고 있나 라벨링된 행렬이다
attributes = pd.read_csv("./list_attr_celeba.csv")
print(attributes.columns)  # 특징 무엇 무엇이 있는지 확인
print(attributes.head())  # 어떤 사진에 어떤 특징이 있나 확인

# 특정 레이블(금발)을 가진 얼굴 데이터 로드
LABEL = "Blond_Hair"
labelled_test = utils.image_dataset_from_directory(
    "./img_align_celeba",
    labels=attributes[LABEL].tolist(),
    color_mode="rgb",
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42,
    validation_split=0.2,
    subset="validation",
    interpolation="bilinear",
)
# 이미지와 라벨의 쌍을 전처리된 이미지와 라벨의 쌍으로 사상(寫像)
labelled = labelled_test.map(lambda x, y: (preprocess(x), y))


# 속성 벡터 찾기
attribute_vec = get_vector_from_label(labelled, vae, Z_DIM, LABEL)
# 이미지에 벡터(속성)을 더하거나 빼보는 작업
add_vector_to_images(labelled, vae, attribute_vec)
# 두 이미지의 특징 벡터 비율을 섞어 디코딩 하여 얼굴을 섞어보는 작업
morph_faces(labelled, vae)