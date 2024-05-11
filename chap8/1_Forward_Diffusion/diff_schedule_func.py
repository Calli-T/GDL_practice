from load_datasets import *

import math
import matplotlib.pyplot as plt


# 선형 확산 과정
def linear_diffusion_schedule(diffusion_times):
    # 최대와 최소값을 정해놓고, B_t값을 diffusion 횟수에 맞춰 선형적으로 올린다
    # a_t는 1 - B_t
    min_rate = 0.0001
    max_rate = 0.02
    betas = min_rate + diffusion_times * (max_rate - min_rate)
    alphas = 1 - betas

    # 축을 따라 텐서 요소의 곱을 구하는 cumprod함수, [a, b, c] -> [a, a * b, a * b * c]
    # https://www.tensorflow.org/api_docs/python/tf/math/cumprod
    # 이는 a_t_bar를 구현한다
    alpha_bars = tf.math.cumprod(alphas)

    # signal_rates는 text1의 sqrt(a_t_bar) * x_0, 원본이미지의 비율
    # noise_rates는 text1의 sqrt(1 - a_t_bar) * ε, 노이즈의 비율
    signal_rates = tf.sqrt(alpha_bars)
    noise_rates = tf.sqrt(1 - alpha_bars)

    # 반환해서 어디에 쓴단말인가?
    return noise_rates, signal_rates


def cosine_diffusion_schedule(diffusion_times):
    # a_t_bar = cos ((t/T) * (math.pi/2))^2 구현
    signal_rates = tf.cos(diffusion_times * math.pi / 2)
    noise_rates = tf.sin(diffusion_times * math.pi / 2)
    return noise_rates, signal_rates


def offset_cosine_diffusion_schedule(diffusion_times):
    # min
    min_signal_rate = 0.02
    max_signal_rate = 0.95

    # 18.19° to 88.85°
    start_angle = tf.acos(max_signal_rate)
    end_angle = tf.acos(min_signal_rate)

    # cosine 18.19° = 0.95002
    # cosine 88.85° = 0.02006
    # diffusion time에 맞춰 조금씩 cos에 들어가는 값이 증가한다
    # 이 구간에서는 x가 작을 수록 cos x가 크다
    # offset이라는 이름답게 최소한의 시작 잡음을 보장한다
    diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

    signal_rates = tf.cos(diffusion_angles)
    noise_rates = tf.sin(diffusion_angles)

    return noise_rates, signal_rates


# t/T는 [0, 1)사이의 값, 0/T부터 t-1/T까지 t개의 값이다
T = 1000
diffusion_times = tf.convert_to_tensor([x / T for x in range(T)])

# 선형, 코사인, 오프셋 코사인 잡음'비' 생성
linear_noise_rates, linear_signal_rates = linear_diffusion_schedule(
    diffusion_times
)
cosine_noise_rates, cosine_signal_rates = cosine_diffusion_schedule(
    diffusion_times
)
(
    offset_cosine_noise_rates,
    offset_cosine_signal_rates,
) = offset_cosine_diffusion_schedule(diffusion_times)


# 이하 순수 출력 구간
def showVarRatio():
    # signal
    plt.plot(
        diffusion_times, linear_signal_rates ** 2, linewidth=1.5, label="linear"
    )
    plt.plot(
        diffusion_times, cosine_signal_rates ** 2, linewidth=1.5, label="cosine"
    )
    plt.plot(
        diffusion_times,
        offset_cosine_signal_rates ** 2,
        linewidth=1.5,
        label="offset_cosine",
    )

    plt.xlabel("t/T", fontsize=12)
    plt.ylabel(r"$\bar{\alpha_t}$ (signal)", fontsize=12)
    plt.legend()
    plt.show()

    # noise
    plt.plot(
        diffusion_times, linear_noise_rates ** 2, linewidth=1.5, label="linear"
    )
    plt.plot(
        diffusion_times, cosine_noise_rates ** 2, linewidth=1.5, label="cosine"
    )
    plt.plot(
        diffusion_times,
        offset_cosine_noise_rates ** 2,
        linewidth=1.5,
        label="offset_cosine",
    )

    plt.xlabel("t/T", fontsize=12)
    plt.ylabel(r"$1-\bar{\alpha_t}$ (noise)", fontsize=12)
    plt.legend()
    plt.show()

# showVarRatio()

# - 2 -
