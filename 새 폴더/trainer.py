import wandb
import segmentation_models_pytorch as smp
from .train_utils import TrainEpoch, ValidEpoch
from .loss import custom_loss, custom_lossv
from .dataloader import Dataset
from .transformations import get_training_augmentation, get_validation_augmentation, get_preprocessing
from .model import Unet
from torchmetrics import StructuralSimilarityIndexMeasure
from torchmetrics import PeakSignalNoiseRatio
import torch
from torch.utils.data import DataLoader

def train(epochs, batch_size, hr_dir, tar_dir, th_dir, hr_val_dir, tar_val_dir, th_val_dir,
          encoder='resnet34', encoder_weights='imagenet', device='cuda', lr=1e-4, beta=1, w_ssim=0.1, w_psnr=0.1):

    # 출력 활성화 함수: tanh → [-1, 1] 범위 출력 (target과 동일한 정규화 범위)
    activation = 'tanh'

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
    loss = custom_loss(batch_size, beta=beta, w_ssim=w_ssim, w_psnr=w_psnr)
    lossV = custom_lossv(w_ssim=w_ssim, w_psnr=w_psnr)

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
    for i in range(0, epochs):
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        print(train_logs)

        # wandb에 학습/검증 지표 기록
        wandb.log({
            'epoch': i + 1,
            't_loss': train_logs['custom_loss'],
            't_ssim': train_logs['ssim'],
            't_psnr': train_logs['psnr'],
            'v_loss': valid_logs['custom_lossv'],
            'v_ssim': valid_logs['ssim'],
            'v_psnr': valid_logs['psnr'],
        })

        # validation SSIM이 개선될 때 모델 저장
        if max_ssim <= valid_logs['ssim']:
            max_ssim = valid_logs['ssim']
            max_psnr = valid_logs['psnr']
            wandb.config.update({'max_ssim': max_ssim, 'max_psnr': max_psnr}, allow_val_change=True)
            torch.save(model.state_dict(), './best_model.pth')
            print('Model saved!')

    print(f'max ssim: {max_ssim} max psnr: {max_psnr}')


def train_model(configs):
    train(configs['epochs'], configs['batch_size'], configs['hr_dir'],
          configs['tar_dir'], configs['th_dir'], configs['hr_val_dir'],
          configs['tar_val_dir'], configs['th_val_dir'], configs['encoder'],
          configs['encoder_weights'], configs['device'], configs['lr'], configs['beta'],
          configs['w_ssim'], configs['w_psnr'])
