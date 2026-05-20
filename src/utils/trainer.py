import segmentation_models_pytorch as smp
from .train_utils import TrainEpoch, ValidEpoch
from .loss import custom_loss, custom_lossv
from .dataloader import Dataset
from .transformations import get_training_augmentation, get_validation_augmentation, get_preprocessing
from .model import Unet
from torchmetrics import StructuralSimilarityIndexMeasure
from torchmetrics import PeakSignalNoiseRatio
import torch
import os
from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time

def train(epochs, batch_size, hr_dir, tar_dir, th_dir, hr_val_dir, tar_val_dir, th_val_dir,
          encoder='resnet34', encoder_weights='imagenet', device='cuda', lr=1e-4, beta=1,
          w_ssim=0.1, w_psnr=0.1, w_L2=0.1, w_L1=0.1, tensorboard_log_dir=None):

    def write_tensorboard_scalars(writer, logs, stage, epoch):
        tag_map = {
            'loss': 'Loss',
            'ssim': 'SSIM',
            'psnr': 'PSNR',
        }

        for log_name, tag_prefix in tag_map.items():
            if log_name in logs:
                writer.add_scalar(f'{tag_prefix}/{stage}', float(logs[log_name]), epoch)

    # 출력 활성화 함수: tanh → [-1, 1] 범위 출력 (target과 동일한 정규화 범위)
    activation = 'tanh'

    if tensorboard_log_dir is None:
        run_name = datetime.now().strftime('%Y%m%d-%H%M%S')
        tensorboard_log_dir = os.path.join('runs', run_name)
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_log_dir)

    model = Unet(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        encoder_depth=5,
        classes=1,
        activation=activation,
        fusion=True,
        contrastive=True,
    )

    # 인코더별 전처리 함수 (ImageNet 정규화 등)
    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weights)

    train_dataset = Dataset(
        hr_dir,
        th_dir,
        tar_dir,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn)
    )
    valid_dataset = Dataset(
        hr_val_dir,
        th_val_dir,
        tar_val_dir,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    # 학습용: Contrastive Loss 포함 / 검증용: Contrastive Loss 제외
    loss = custom_loss(batch_size, beta=beta, w_ssim=w_ssim, w_psnr=w_psnr, w_L2 = w_L2, w_L1 = w_L1)
    lossV = custom_lossv(w_ssim=w_ssim, w_psnr=w_psnr, w_L2 = w_L2, w_L1 = w_L1)

    # 평가 메트릭: SSIM (구조적 유사도), PSNR (신호 대 잡음비)
    ssim_metric = StructuralSimilarityIndexMeasure()
    psnr_metric = PeakSignalNoiseRatio()
    psnr_metric.__name__ = 'psnr'
    ssim_metric.__name__ = 'ssim'
    metrics = [ssim_metric, psnr_metric]

    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=lr),
    ])

    train_epoch = TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=device,
        verbose=True,
        contrastive=True
    )
    valid_epoch = ValidEpoch(
        model,
        loss=lossV,
        metrics=metrics,
        device=device,
        verbose=True,
        contrastive=True
    )

    max_ssim = 0
    max_psnr = 0

    # 모델 save 폴더
    model_dir = f'models/{time.strftime("%Y%m%d-%H%M%S")}_model'
    os.makedirs(model_dir, exist_ok=True)

    try:
        for i in range(0, epochs):
            print('\nEpoch: {}'.format(i))
            train_logs = train_epoch.run(train_loader)
            valid_logs = valid_epoch.run(valid_loader)
            print(train_logs)

            write_tensorboard_scalars(writer, train_logs, 'train', i)
            write_tensorboard_scalars(writer, valid_logs, 'valid', i)
            writer.flush()

            # validation SSIM이 개선될 때 모델 저장
            if max_ssim <= valid_logs['ssim']:
                max_ssim = valid_logs['ssim']
                max_psnr = valid_logs['psnr']

                torch.save(model.state_dict(), os.path.join(model_dir, 'best_model.pth'))
                print('Model saved!')
    finally:
        writer.close()

    print(f'max ssim: {max_ssim} max psnr: {max_psnr}')
    print(f'TensorBoard logs saved to: {tensorboard_log_dir}')


def train_model(configs):
    train(configs['epochs'], configs['batch_size'], configs['hr_dir'],
          configs['tar_dir'], configs['th_dir'], configs['hr_val_dir'],
          configs['tar_val_dir'], configs['th_val_dir'], configs['encoder'],
          configs['encoder_weights'], configs['device'], configs['lr'], configs['beta'],
          configs['w_ssim'], configs['w_psnr'], configs['w_L2'], configs['w_L1'], configs.get('tensorboard_log_dir'))
