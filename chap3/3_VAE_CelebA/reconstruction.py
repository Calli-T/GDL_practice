import numpy as np

from train import *

from scipy.stats import norm
import matplotlib.pyplot as plt

# 모델 가져오기
vae = tf.keras.models.load_model("./models/vae")
en = tf.keras.models.load_model("./models/encoder")
de = tf.keras.models.load_model("./models/decoder")

# 테스트 세트에서 일부분을 선택
batches_to_predict = 1
example_images = np.array(list(train.take(batches_to_predict).get_single_element()))

# 오토인코더 predict & 출력
z_mean, z_log_var, reconstructions = vae.predict(example_images)
print("실제 얼굴")
plt.figure(figsize=(10, 10))
for idx, img in enumerate(example_images[:10]):
    plt.subplot(10, 10, idx + 1)
    plt.imshow(img, cmap="gray")
plt.show()
print("재구성")
plt.figure(figsize=(10, 10))
for idx, img in enumerate(reconstructions[:10]):
    plt.subplot(10, 10, idx + 1)
    plt.imshow(img, cmap="gray")
plt.show()

# 잠재공간 분포 확인
_, _, z = vae.encoder.predict(example_images)

x = np.linspace(-3, 3, 100)
fig = plt.figure(figsize=(20, 5))
fig.subplots_adjust(hspace=0.6, wspace=0.4)

for i in range(50):
    ax = fig.add_subplot(5, 10, i + 1)
    ax.hist(z[:, i], density=True, bins=20)
    ax.axis("off")
    ax.text(0.5, -0.35, str(i), fontsize=10, ha="center", transform=ax.transAxes)
    ax.plot(x, norm.pdf(x))
plt.show()

# - 생성과 출력 -
gw, gh = (10, 3)
z_sample = np.random.normal(size=(gw * gh, Z_DIM))

reconstructions = de.predict(z_sample)

fig = plt.figure(figsize=(18, 5))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i in range(gh * gw):
    ax = fig.add_subplot(gh, gw, i + 1)
    ax.axis("off")
    ax.imshow(reconstructions[i, :, :])
plt.show()
# - 4.1 -
