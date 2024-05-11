from class_ddm import *

from tensorflow.keras import optimizers, losses, callbacks

# hyper
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 50
PLOT_DIFFUSION_STEPS = 20

ddm = DiffusionModel()
ddm.normalizer.adapt(train)

# 체크포인트 불러오기 설정
LOAD_MODEL = True

if LOAD_MODEL:
    ddm.built = True
    ddm.load_weights("./checkpoint/checkpoint.ckpt")

ddm.compile(
    optimizer=optimizers.experimental.AdamW(
        learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    ),
    loss=losses.mean_absolute_error,
)

# 훈련을 실행하고 생성된 이미지를 주기적으로 출력합니다.
model_checkpoint_callback = callbacks.ModelCheckpoint(
    filepath="./checkpoint/checkpoint.ckpt",
    save_weights_only=True,
    save_freq="epoch",
    verbose=0,
)

tensorboard_callback = callbacks.TensorBoard(log_dir="./logs")


class ImageGenerator(callbacks.Callback):
    def __init__(self, num_img):
        self.num_img = num_img

    def on_epoch_end(self, epoch, logs=None):
        generated_images = self.model.generate(
            num_images=self.num_img,
            diffusion_steps=PLOT_DIFFUSION_STEPS,
        ).numpy()
        display(
            generated_images,
            save_to="./output/generated_img_%03d.png" % (epoch),
        )


image_generator_callback = ImageGenerator(num_img=10)

# if not len(tf.config.list_physical_devices('GPU')) > 0:
#     raise Exception('tf.config.list_physical_devices(\'GPU\') returns empty list; no GPUs found')
# print(tf.config.list_physical_devices())
#
# with tf.device("/GPU:0"):
#     ddm.fit(
#         train,
#         epochs=EPOCHS,
#         callbacks=[
#             model_checkpoint_callback,
#             tensorboard_callback,
#             image_generator_callback,
#         ],
#     )


# 컴파일, 콜백, 주기적으로 이미지 생성, 모델저장, 장치설정, 학습
# - 6 -
