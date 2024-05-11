from residual_unet import *

from tensorflow.keras import metrics

# hyper
EMA = 0.999


# 텐서플로우 문법들로 이루어진 클래스
class DiffusionModel(models.Model):
    def __init__(self):
        super().__init__()

        # 정규화 클래스, 신경망, ema신경망, 확산 스케줄...
        self.normalizer = layers.Normalization()
        self.network = unet
        self.ema_network = models.clone_model(self.network)
        self.diffusion_schedule = offset_cosine_diffusion_schedule

    #
    def compile(self, **kwargs):
        super().compile(**kwargs)
        self.noise_loss_tracker = metrics.Mean(name="n_loss")  # 평가 척도 관련 코드인듯

    @property
    def metrics(self):
        return [self.noise_loss_tracker]

    '''
    비정규화, 신경망에서 출력된 이미지는 표준 정규 분포를 따른다
    여기에 훈련 데이터셋에서 계산한 평균과 분산으로 비정규화한다
    아마도 ??? 추측컨데, 가법성을 위해 이미지의 픽셀에 대해 정규화 했으니
    반대로 비정규화해서 값에 맞도록 처리해주는 과정도 있어야 이미지로 보이게 만들어지는듯
    '''
    def denormalize(self, images):
        images = self.normalizer.mean + images * self.normalizer.variance ** 0.5
        return tf.clip_by_value(images, 0.0, 1.0)  # [0.0, 1.0]구간으로 자른다. 픽셀 범위 맞추는듯

    # 잡음 제거 함수, 훈련 중에는 원본 신경망을 쓰고 아닐 때는 ema 신경망을 사용한다
    # 잡음비(잡음의 표준편차이기도 하다)와 이미지를 받아
    # u-net으로 이미지의 잡음을 예측한다
    # 이를 신호비(신호의 표준편차이기도 하다)와 잡음낀 이미지로 역산해 예측된 원본 이미지를 만들어낸다
    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        if training:
            network = self.network
        else:
            network = self.ema_network
        pred_noises = network(
            [noisy_images, noise_rates ** 2], training=training
        )
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates

        return pred_noises, pred_images

    def reverse_diffusion(self, initial_noise, diffusion_steps):
        num_images = initial_noise.shape[0]  # [batch_size, h, w, c]에서 첫 번째, 배치 개수
        step_size = 1.0 / diffusion_steps  # 전체 스탭 T를 고정적으로 1로 간주하고, 1 step당 크기를 계산
        current_images = initial_noise  # 노이즈 제거 이전의 역방향 확산 과정의 시작점 x_t
        for step in range(diffusion_steps):  # (임의로 지정되어 고정된) 역방향 확산 과정의 단계수만큼 반복
            '''
            [0, 1]사이의 스탭 t를 구하고
            구한 t로 확산 스케줄을 구한다
            shape이 [10, 1, 1, 1]인걸로 봐서는 reverse_diffusion 함수 자체가 이미지 생성 때만 쓰는듯
            '''
            diffusion_times = tf.ones((num_images, 1, 1, 1)) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)

            '''
            잡음제거 함수로 예측된 이미지 x_0(pred_images)를 받아 온다
            단계 t의 바로 다음 단계 t-1를 구하고
            이를 스케줄 함수에 넣어 다음 신호비(= 신호의 표준 편차)와 잡음비(= 잡음의 표준 편차)를 가져온다
            예측된 이미지 x_0와 t-1단계의 신호비, 잡음비, 예측된 잡음(예측 이미지 가져올 때 같이 가져옴)을 가지고 이미지 x_t-1를 예측해낸다
            이 예제는 샘플링에 무작위성이 '없다' 원래는 x_t-1를 만드는 과정에서 무작위 잡음을 추가해야 한다
            '''
            pred_noises, pred_images = self.denoise(
                current_images, noise_rates, signal_rates, training=False
            )
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(
                next_diffusion_times
            )
            current_images = (
                    next_signal_rates * pred_images + next_noise_rates * pred_noises
            )

        # 최종 예측 이미지를 반환
        return pred_images

    '''
    생성함수, 무작위 잡음을 뽑아 역방향 확산 처리하고, 이를 비정규화하여 그림으로 만들어낸다
    '''
    def generate(self, num_images, diffusion_steps, initial_noise=None):
        if initial_noise is None:
            initial_noise = tf.random.normal(
                shape=(num_images, IMAGE_SIZE, IMAGE_SIZE, 3)
            )
        generated_images = self.reverse_diffusion(
            initial_noise, diffusion_steps
        )
        generated_images = self.denormalize(generated_images)
        return generated_images


    def train_step(self, images):
        # 이미지를 정규화, 잡음은 이미지 크기에 맞게 생성
        images = self.normalizer(images, training=True)
        noises = tf.random.normal(shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3))

        '''
        책에는 (균등 분포를 따르는) 무작위 확산 시간을 샘플링 한다고 하는데, 이게 무슨 소리인지 모르겠다
        일단 그걸로 코사인 확산 스케줄을 만들어낸다
        아마도, 각각의 x_t에 적용되는 a_t_bar 즉 분산을 무작위로 뽑아내는 작업인듯
        1-a_t_bar와 a_t_bar가 모두 사용되므로, 0과 1사이의 균등분포를 따르는듯
        '''
        diffusion_times = tf.random.uniform(
            shape=(BATCH_SIZE, 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        noisy_images = signal_rates * images + noise_rates * noises # 정방향 확산 과정은 한 번에 바로 적용한다

        with tf.GradientTape() as tape:
            # 잡음 이미지에서 잡음을 구분하도록 훈련합니다.
            # unet으로 예측
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, training=True
            )

            noise_loss = self.loss(noises, pred_noises)  # 손실 함수에서 실제 이미지랑 노이즈낀 실제이미지에서 역방향 확산 예측한걸 비교,

        # 아무튼 경사하강법, 문법은 모름
        gradients = tape.gradient(noise_loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(
            zip(gradients, self.network.trainable_weights)
        )

        self.noise_loss_tracker.update_state(noise_loss)

        # ema 신경망에 카피
        for weight, ema_weight in zip(
                self.network.weights, self.ema_network.weights
        ):
            ema_weight.assign(EMA * ema_weight + (1 - EMA) * weight)

        return {m.name: m.result() for m in self.metrics}

    # 위랑 똑같은데 경사하강은 없고 ema 모델로 처리하며, 점수를 예측한것들의 평균으로 매긴다
    def test_step(self, images):
        images = self.normalizer(images, training=False)
        noises = tf.random.normal(shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3))
        diffusion_times = tf.random.uniform(
            shape=(BATCH_SIZE, 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        noisy_images = signal_rates * images + noise_rates * noises
        pred_noises, pred_images = self.denoise(
            noisy_images, noise_rates, signal_rates, training=False
        )
        noise_loss = self.loss(noises, pred_noises)
        self.noise_loss_tracker.update_state(noise_loss)

        return {m.name: m.result() for m in self.metrics}

# - 5 -
