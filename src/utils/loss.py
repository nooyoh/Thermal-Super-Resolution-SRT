import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import StructuralSimilarityIndexMeasure
from torchmetrics import PeakSignalNoiseRatio
from segmentation_models_pytorch.utils import base

def device_as(t1, t2):
    # t1을 t2와 같은 디바이스(CPU/GPU)로 이동
    return t1.to(t2.device)


class ContrastiveLoss(nn.Module):
    # SimCLR 방식의 InfoNCE Loss
    # VIS 인코더와 Thermal 인코더의 feature가 같은 장면이면 가깝게, 다른 장면이면 멀어지도록 학습
    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        # 자기 자신과의 유사도를 제외하기 위한 마스크 (대각선 제거)
        self.mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float()

    def calc_similarity_batch(self, a, b):
        # a와 b를 합쳐 2N개의 임베딩에 대한 코사인 유사도 행렬 계산
        representations = torch.cat([a, b], dim=0)
        return F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

    def forward(self, proj_1, proj_2):
        # proj_1: VIS 인코더 feature [batch, embedding_dim]
        # proj_2: Thermal 인코더 feature [batch, embedding_dim]
        # 같은 인덱스끼리 positive pair (SimCLR의 z_i, z_j)
        batch_size = proj_1.shape[0]
        z_i = F.normalize(proj_1, p=2, dim=1)
        z_j = F.normalize(proj_2, p=2, dim=1)

        similarity_matrix = self.calc_similarity_batch(z_i, z_j)

        # positive pair의 유사도만 추출
        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)
        denominator = device_as(self.mask, similarity_matrix) * torch.exp(similarity_matrix / self.temperature)

        all_losses = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(all_losses) / (2 * self.batch_size)
        return loss


class custom_loss(base.Loss):
    # 학습용 손실 함수: MSE + w_ssim*SSIM항 + w_psnr*PSNR항 + beta*ContrastiveLoss
    def __init__(self, batch_size, beta, w_ssim=0.1, w_psnr=0.1, w_L1=0.1, w_L2 = 0.1):
        super().__init__()
        self.ssim = StructuralSimilarityIndexMeasure()
        self.mse = nn.MSELoss()
        # L1 Loss 추가
        self.L1 = nn.L1Loss()
        self.contrast = ContrastiveLoss(batch_size)
        self.psnr = PeakSignalNoiseRatio()
        self.beta = beta
        self.w_ssim = w_ssim
        self.w_psnr = w_psnr
        self.w_L1 = w_L1
        self.w_L2 = w_L2

    def forward(self, y_pr, y_gt, ft1=None, ft2=None):
        l1_loss = self.L1(y_pr, y_gt)
        l2_loss = self.mse(y_pr, y_gt)
        y = 1 - self.ssim(y_pr, y_gt)
        z = self.contrast(ft1, ft2)
        # PSNR은 높을수록 좋으므로 40 기준으로 정규화하여 손실로 변환
        p = (1 - self.psnr(y_pr, y_gt) / 40)
        return l1_loss * self.w_L1 + l2_loss * self.w_L2 + self.w_ssim * y + self.w_psnr * p + (z * self.beta)


class custom_lossv(base.Loss):
    # 검증용 손실 함수: Contrastive Loss 제외 (검증 시 feature 정렬 불필요)
    # MSE + w_ssim*SSIM항 + w_psnr*PSNR항
    def __init__(self, w_ssim=0.1, w_psnr=0.1, w_L1=0.1, w_L2=0.1):
        super().__init__()
        self.ssim = StructuralSimilarityIndexMeasure()
        self.mse = nn.MSELoss()
        self.L1 = nn.L1Loss()
        self.psnr = PeakSignalNoiseRatio()
        self.w_ssim = w_ssim
        self.w_psnr = w_psnr
        self.w_L1 = w_L1
        self.w_L2 = w_L2

    def forward(self, y_pr, y_gt, ft1=None, ft2=None):
        l1_loss = self.L1(y_pr, y_gt)
        l2_loss = self.mse(y_pr, y_gt)
        y = 1 - self.ssim(y_pr, y_gt)
        p = (1 - self.psnr(y_pr, y_gt) / 40)

        return l1_loss * self.w_L1 + l2_loss * self.w_L2 + self.w_ssim * y + self.w_psnr * p
        