import numpy as np
import matplotlib.pyplot as plt

def get_vector_from_label(data, vae, embedding_dim, label):
    # 양은 해당 속성을 가진 이미지의 특징 벡터, 음은 해당 속성이 이미지의 특징 벡터를 의미

    # 현재 양의 벡터 합, 현재 양의 벡터 수, 현재 양의 벡터 평균
    current_sum_POS = np.zeros(shape=embedding_dim, dtype="float32")
    current_n_POS = 0
    current_mean_POS = np.zeros(shape=embedding_dim, dtype="float32")

    # 현재 음의 벡터 합, 현재 음의 벡터 수, 현재 음의 벡터 평균
    current_sum_NEG = np.zeros(shape=embedding_dim, dtype="float32")
    current_n_NEG = 0
    current_mean_NEG = np.zeros(shape=embedding_dim, dtype="float32")

    # 현재 벡터, 현재 거리
    current_vector = np.zeros(shape=embedding_dim, dtype="float32")
    current_dist = 0

    print("label: " + label)
    print("images : POS move : NEG move :distance : 𝛥 distance")
    while current_n_POS < 10000:
        # 배치 하나를 가져옴
        batch = list(data.take(1).get_single_element())
        im = batch[0]
        attribute = batch[1]
        # im은 이미지 (1배치 128개, 사이즈64x64, 채널 rgb 3 -> [128, 64, 64, 3])
        # attribute는 해당 속성을 가지고 있으면 1 없으면 -1 -> [128]

        # 해당 배치를 모조리 특징 벡터로 변환
        _, _, z = vae.encoder.predict(np.array(im), verbose=0)

        # attribute값을 가지고 특징 벡터를 fancy indexing
        z_POS = z[attribute == 1]
        z_NEG = z[attribute == -1]

        # 양의 벡터에 대해 합을 갱신하고, 벡터 수를 갱신하고, 평균을 구하고
        # 양의 벡터에 대해 새 평균과 벡터와 옛 평균 벡터로 거리(노름)를 구한다 이건 해당 속성이 있는 이미지의 특징벡터가 얼마나 변했는가를 의미한다
        if len(z_POS) > 0:
            current_sum_POS = current_sum_POS + np.sum(z_POS, axis=0)
            current_n_POS += len(z_POS)
            new_mean_POS = current_sum_POS / current_n_POS
            movement_POS = np.linalg.norm(new_mean_POS - current_mean_POS)

        # 음의 벡터에 대해 합을 갱신하고, 벡터 수를 갱신하고, 평균을 구하고
        # 음의 벡터에 대해 새 평균과 벡터와 옛 평균 벡터로 거리(노름)를 구한다 이건 해당 속성이 없는 이미지의 특징벡터가 얼마나 변했는가를 의미한다
        if len(z_NEG) > 0:
            current_sum_NEG = current_sum_NEG + np.sum(z_NEG, axis=0)
            current_n_NEG += len(z_NEG)
            new_mean_NEG = current_sum_NEG / current_n_NEG
            movement_NEG = np.linalg.norm(new_mean_NEG - current_mean_NEG)

        # 양의 벡터에서 음의 벡터를 뺀, 즉 속성을 가진 평균과 그렇지 않은 평균의 차를 구한다
        # 노름으로 벡터의 길이를 구한다
        # 벡터의 길이 변화를 구한다?
        current_vector = new_mean_POS - new_mean_NEG
        new_dist = np.linalg.norm(current_vector)
        dist_change = new_dist - current_dist

        print(
            str(current_n_POS)
            + "    : "
            + str(np.round(movement_POS, 3))
            + "    : "
            + str(np.round(movement_NEG, 3))
            + "    : "
            + str(np.round(new_dist, 3))
            + "    : "
            + str(np.round(dist_change, 3))
        )

        # 음양의 평균 벡터와 벡터의 길이를 최신화한다
        current_mean_POS = np.copy(new_mean_POS)
        current_mean_NEG = np.copy(new_mean_NEG)
        current_dist = np.copy(new_dist)

        # 음양 모두의 특징 벡터가 크게 안바뀌었다면
        # 속성을 나타내는 특징 벡터를 단위 벡터로 만든 다음 반환할 준비를 한다
        if np.sum([movement_POS, movement_NEG]) < 0.08:
            current_vector = current_vector / current_dist
            print("Found the " + label + " vector")
            break

    return current_vector

def add_vector_to_images(data, vae, feature_vec):
    n_to_show = 5
    factors = [-4, -3, -2, -1, 0, 1, 2, 3, 4]

    example_batch = list(data.take(1).get_single_element())
    example_images = example_batch[0]

    _, _, z_points = vae.encoder.predict(example_images, verbose=0)

    fig = plt.figure(figsize=(18, 10))

    counter = 1

    for i in range(n_to_show):
        img = example_images[i]
        sub = fig.add_subplot(n_to_show, len(factors) + 1, counter)
        sub.axis("off")
        sub.imshow(img)

        counter += 1

        for factor in factors:
            changed_z_point = z_points[i] + feature_vec * factor
            changed_image = vae.decoder.predict(
                np.array([changed_z_point]), verbose=0
            )[0]

            sub = fig.add_subplot(n_to_show, len(factors) + 1, counter)
            sub.axis("off")
            sub.imshow(changed_image)

            counter += 1

    plt.show()


def morph_faces(data, vae):
    factors = np.arange(0, 1, 0.1)

    example_batch = list(data.take(1).get_single_element())[:2]
    example_images = example_batch[0]
    _, _, z_points = vae.encoder.predict(example_images, verbose=0)

    fig = plt.figure(figsize=(18, 8))

    counter = 1

    img = example_images[0]
    sub = fig.add_subplot(1, len(factors) + 2, counter)
    sub.axis("off")
    sub.imshow(img)

    counter += 1

    for factor in factors:
        changed_z_point = z_points[0] * (1 - factor) + z_points[1] * factor
        changed_image = vae.decoder.predict(
            np.array([changed_z_point]), verbose=0
        )[0]
        sub = fig.add_subplot(1, len(factors) + 2, counter)
        sub.axis("off")
        sub.imshow(changed_image)

        counter += 1

    img = example_images[1]
    sub = fig.add_subplot(1, len(factors) + 2, counter)
    sub.axis("off")
    sub.imshow(img)

    plt.show()
