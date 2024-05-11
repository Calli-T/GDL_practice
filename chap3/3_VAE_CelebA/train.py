from set_VAE import *

from tensorflow import optimizers
from keras import callbacks

# parameters
LEARNING_RATE = 0.0005
EPOCHS = 10
LOAD_MODEL = False

# get VAE
en, de = get_EnDe()
vae = VAE(en, de)

# compile
optimizer = optimizers.Adam(learning_rate=LEARNING_RATE)
vae.compile(optimizer=optimizer)

# checkpoint callback & tensorboard callback
model_checkpoint_callback = callbacks.ModelCheckpoint(
    filepath="./checkpoint",
    save_weights_only=False,
    save_freq="epoch",
    monitor="loss",
    mode="min",
    save_best_only=True,
    verbose=0,
)

tensorboard_callback = callbacks.TensorBoard(log_dir="./logs")

# # 필요한 경우 이전 가중치 로드
# if LOAD_MODEL:
#     vae.load_weights("./models/vae")
#     tmp = vae.predict(train.take(1))

# 이미지 생성기가 대체 왜 콜백에?
'''
class ImageGenerator(callbacks.Callback):
    def __init__(self, num_img, latent_dim):
        self.num_img = num_img
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.normal(
            shape=(self.num_img, self.latent_dim)
        )
        generated_images = self.model.decoder(random_latent_vectors)
        generated_images *= 255
        generated_images.numpy()
        for i in range(self.num_img):
            img = utils.array_to_img(generated_images[i])
            img.save("./output/generated_img_%03d_%d.png" % (epoch, i))
'''


def train_and_save():
    vae.fit(
        train,
        epochs=EPOCHS,
        callbacks=[
            model_checkpoint_callback,
            tensorboard_callback,
            # ImageGenerator(num_img=10, latent_dim=Z_DIM),
        ],
    )

    # 최종 모델 저장
    vae.save("./models/vae")
    vae.encoder.save("./models/encoder")
    vae.decoder.save("./models/decoder")

# - 3 -
