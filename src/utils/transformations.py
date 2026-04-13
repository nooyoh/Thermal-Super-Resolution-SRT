import albumentations as albu

def get_training_augmentation():
    # 학습 데이터 증강: 리사이즈 + 좌우/상하 반전
    # VIS와 Thermal 이미지 해상도가 달라 is_check_shapes=False 설정
    train_transform = [
        albu.Resize(480, 640, always_apply=True),
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.PadIfNeeded(min_height=480, min_width=640, always_apply=True, border_mode=0),
    ]
    return albu.Compose(train_transform, additional_targets={'image1': 'mask'}, is_check_shapes=False)

def get_validation_augmentation():
    # 검증 데이터 전처리: 리사이즈만 적용
    test_transform = [
        albu.Resize(480, 640, always_apply=True),
        albu.PadIfNeeded(480, 640)
    ]
    return albu.Compose(test_transform, additional_targets={'image1': 'mask'}, is_check_shapes=False)


def to_tensor(x, **kwargs):
    # (H, W, C) → (C, H, W) 변환 및 float32 변환
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn):
    # 인코더별 정규화 및 텐서 변환
    # preprocessing_fn: 인코더(예: ResNet34)에 맞는 ImageNet 정규화 함수
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)
