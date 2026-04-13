import matplotlib.pyplot as plt
import os

def normalize_data(data):
    # [0, 1] 범위를 tanh 출력 범위인 [-1, 1]로 변환
    data *= 2
    data -= 1
    return data

def unnormalize_data(data):
    # [-1, 1] 범위를 [0, 1]로 역변환
    data += 1
    data /= 2
    return data

def visualize(**images):
    # 여러 이미지를 한 행에 나란히 시각화
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()

def list_img(dir1):
    # 디렉토리 내 이미지 파일을 정렬하여 전체 경로 리스트로 반환
    lst = []
    for root, dirs, files in os.walk(dir1):
        lst.extend(files)
    lst = sorted(lst)
    for x in range(len(lst)):
        lst[x] = dir1 + '/' + lst[x]
    return lst
