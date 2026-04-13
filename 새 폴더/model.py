from typing import Optional, Union, List
import torch
import torch.nn as nn
from segmentation_models_pytorch.base.modules import Activation
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder

def initialize_decoder(module):
    # 디코더 가중치 초기화: Conv2d는 Kaiming, BatchNorm은 1/0, Linear는 Xavier
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

def initialize_head(module):
    # 출력 헤드 가중치 초기화: Xavier uniform
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class SegmentationHead(nn.Sequential):
    # 디코더 출력을 최종 예측 맵으로 변환하는 헤드 (Conv → Upsample → Activation)
    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        activation = Activation(activation)
        super().__init__(conv2d, upsampling, activation)

class SegmentationModel(torch.nn.Module):
    def initialize(self):
        initialize_decoder(self.decoder)
        initialize_head(self.segmentation_head)

    def check_input_shape(self, x):
        # 입력 이미지 크기가 encoder의 output_stride 배수인지 확인
        h, w = x.shape[-2:]
        output_stride = self.encoder.output_stride
        if h % output_stride != 0 or w % output_stride != 0:
            new_h = (h // output_stride + 1) * output_stride if h % output_stride != 0 else h
            new_w = (w // output_stride + 1) * output_stride if w % output_stride != 0 else w
            raise RuntimeError(
                f"Wrong input shape height={h}, width={w}. Expected image height and width "
                f"divisible by {output_stride}. Consider pad your images to shape ({new_h}, {new_w})."
            )

    def forward(self, x, y=None):
        # x: VIS HR 이미지, y: Thermal LR 이미지
        self.check_input_shape(x)

        features = self.encoder(x)
        if self.fusion == True:
            features1 = self.encoder2(y)

            # Contrastive Loss 계산을 위해 각 인코더의 마지막 feature 저장
            f1 = features[-1]
            f2 = features1[-1]

            # 두 인코더의 feature를 element-wise max로 융합
            for ind in range(len(features)):
                features[ind] = torch.maximum(features[ind], features1[ind])

        decoder_output = self.decoder(features)
        masks = self.segmentation_head(decoder_output)

        if self.contrastive_head1 is not None:
            # Contrastive Loss용 projection head를 통해 64차원 임베딩으로 압축
            f1 = self.contrastive_head1(f1)
            f2 = self.contrastive_head2(f2)
            return masks, f1, f2
        return masks

    @torch.no_grad()
    def predict(self, x, y=None):
        # 추론 전용 메서드 (gradient 계산 없음)
        if self.training:
            self.eval()
        if self.contrastive_head1 is not None:
            x, _, _ = self.forward(x, y)
            return x
        if y is not None:
            x = self.forward(x, y)
            return x
        x = self.forward(x)
        return x

class Unet(SegmentationModel):
    # Dual Encoder UNet
    # encoder: VIS HR 이미지용, encoder2: Thermal LR 이미지용
    # 두 인코더의 feature를 element-wise max로 융합 후 단일 디코더로 복원
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        fusion: bool = True,
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        contrastive: bool = False,
    ):
        super().__init__()
        self.fusion = fusion

        # VIS 이미지용 인코더
        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )
        # Thermal 이미지용 인코더 (동일 구조, 독립 가중치)
        self.encoder2 = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = UnetDecoder(
            encoder_channels=(self.encoder.out_channels),
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_norm=decoder_use_batchnorm,
            add_center_block=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        # 출력 채널 1개 (Thermal HR 그레이스케일), 활성화 함수 tanh
        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        if contrastive:
            # 각 인코더의 마지막 feature를 64차원으로 압축하는 projection head
            # Contrastive Loss 계산에 사용
            self.contrastive_head1 = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(in_features=self.encoder.out_channels[-1], out_features=512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Linear(in_features=512, out_features=64),
                nn.BatchNorm1d(64),
            )
            self.contrastive_head2 = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(in_features=self.encoder.out_channels[-1], out_features=512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Linear(in_features=512, out_features=64),
                nn.BatchNorm1d(64),
            )

        self.name = "u-{}".format(encoder_name)
        self.initialize()
