# Thermal Super Resolution

가시광선(VIS) 고해상도 이미지를 가이드로 활용해 열화상(Thermal) 저해상도 이미지를 고해상도로 복원하는 딥러닝 모델입니다.

## 요구 사항

- Python 3.11
- CUDA 지원 GPU 권장

## 설치

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install segmentation-models-pytorch==0.5.0
pip install albumentations==2.0.8
pip install torchmetrics wandb opencv-python matplotlib
```

## 데이터셋 구조

```
Dataset/
└── train_val/
    ├── training/
    │   ├── train_VIS_HR/                       # 가시광선 HR 이미지
    │   ├── train_input_THER_LR_bicubic/X8/     # 열화상 LR 이미지 (1/8 다운샘플)
    │   └── train_output_gt_THER_HR/            # 열화상 HR 정답 이미지
    └── validation/
        ├── valid_VIS_HR/
        ├── valid_input_THER_LR_bicubic/X8/
        └── valid_output_gt_THER_HR/
```

## 실행

```bash
# 기본 학습
python -m src.train

# 동작 확인 (1 에폭, 빠른 테스트)
python -m src.train --test

# 하이퍼파라미터 직접 지정
python -m src.train --epochs 100 --batch_size 8 --lr 1e-4 --beta 1 --w_ssim 0.1 --w_psnr 0.1
```

## 주요 인자

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--epochs` | 250 | 학습 에폭 수 |
| `--batch_size` | 8 | 배치 크기 |
| `--lr` | 1e-4 | 학습률 |
| `--beta` | 1 | Contrastive Loss 가중치 |
| `--w_ssim` | 0.1 | SSIM Loss 가중치 |
| `--w_psnr` | 0.1 | PSNR Loss 가중치 |
| `--encoder` | resnet34 | 인코더 종류 |
| `--test` | - | 빠른 동작 확인 모드 |

*모든 정보는 언제든지 수정될 수 있습니다*