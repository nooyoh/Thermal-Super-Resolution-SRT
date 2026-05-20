import argparse
from .utils.trainer import train_model


def main(args):
    # 학습에 사용할 설정값을 딕셔너리로 묶어 trainer에 전달
    config = {
        'hr_dir': args.hr_dir,
        'th_dir': args.th_dir,
        'tar_dir': args.tar_dir,
        'hr_val_dir': args.hr_val_dir,
        'th_val_dir': args.th_val_dir,
        'tar_val_dir': args.tar_val_dir,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'device':args.device,
        'encoder': args.encoder,
        'encoder_weights': args.encoder_weights,
        'lr': args.lr,
        'beta': args.beta,
        'w_ssim': args.w_ssim,
        'w_psnr': args.w_psnr,
        'w_L1': args.w_L1,
        'w_L2': args.w_L2
    }

    train_model(config)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # 학습 데이터 경로
    parser.add_argument('--hr_dir', type=str, required=False, default='./Dataset/train_val/training/train_VIS_HR')
    parser.add_argument('--th_dir', type=str, required=False, default='./Dataset/train_val/training/train_input_THER_LR_bicubic/X8')
    parser.add_argument('--tar_dir', type=str, required=False, default='./Dataset/train_val/training/train_output_gt_THER_HR')
    # 검증 데이터 경로
    parser.add_argument('--hr_val_dir', type=str, required=False, default='./Dataset/train_val/validation/valid_VIS_HR')
    parser.add_argument('--th_val_dir', type=str, required=False, default='./Dataset/train_val/validation/valid_input_THER_LR_bicubic/X8')
    parser.add_argument('--tar_val_dir', type=str, required=False, default='./Dataset/train_val/validation/valid_output_gt_THER_HR')
    # 학습 하이퍼파라미터
    parser.add_argument('--batch_size', type=int, required=False, default=8)
    parser.add_argument('--epochs', type=int, required=False, default=250)
    parser.add_argument('--device', type=str, required=False, default='cuda')
    parser.add_argument('--encoder', type=str, required=False, default='resnet34')
    parser.add_argument('--encoder_weights', type=str, required=False, default='imagenet')
    parser.add_argument('--lr', type=float, required=False, default=1e-4)
    # Loss 가중치 (손실 함수: MSE + w_ssim*SSIM + w_psnr*PSNR + beta*ContrastiveLoss)
    parser.add_argument('--beta', type=float, required=False, default=1)
    parser.add_argument('--w_ssim', type=float, required=False, default=0.1)
    parser.add_argument('--w_psnr', type=float, required=False, default=0.1)
    parser.add_argument('--w_L1', type=float, required=False, default=0.1)
    parser.add_argument('--w_L2', type=float, required=False, default=0.1)
    # 코드 동작 확인용: epochs=1, batch_size=2로 빠르게 실행
    parser.add_argument('--test', action='store_true', help='Run 1 epoch with batch_size=2 to verify code')
    arguments = parser.parse_args()
    if arguments.test:
        arguments.epochs = 1
        arguments.batch_size = 2
    main(arguments)
