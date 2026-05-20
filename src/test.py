import argparse
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from .utils.model import Unet
    from .utils.transformations import (
        get_preprocessing,
        get_validation_augmentation,
    )
except ImportError:
    from utils.model import Unet
    from utils.transformations import (
        get_preprocessing,
        get_validation_augmentation,
    )


def resolve_existing_path(*candidates: Path) -> Path:
    # 후보 경로들 중 실제 존재하는 첫 번째 경로를 반환 (없으면 첫 번째 후보 반환)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def extract_sample_key(path: Path) -> str:
    # VIS-Thermal 매칭용 키 추출 (_vis, _th 접미사 제거)
    # 예: "0001_vis.png" → "0001", "0001_th.png" → "0001"
    stem = path.stem
    for suffix in ("_vis", "_th"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def tensor_to_uint8_image(tensor: torch.Tensor) -> np.ndarray:
    # 모델 출력 텐서 [-1, 1] 범위(tanh) → uint8 이미지 [0, 255]로 변환
    image = tensor.detach().cpu().squeeze().numpy()
    image = np.clip((image + 1.0) / 2.0, 0.0, 1.0)
    image = (image * 255.0).round().astype(np.uint8)
    return image


class ThermalTestDataset(Dataset):
    # 추론용 데이터셋: VIS HR과 Thermal LR 페어를 키로 매칭하여 로드
    def __init__(self, visible_dir: Path, thermal_dir: Path, encoder: str, encoder_weights: str):
        self.visible_dir = visible_dir
        self.thermal_dir = thermal_dir
        self.samples = self._build_samples()
        # 학습 때와 동일한 인코더 전처리 함수 사용 (ImageNet 정규화 등)
        preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weights)
        self.augmentation = get_validation_augmentation()
        self.preprocessing = get_preprocessing(preprocessing_fn)

    def _build_samples(self):
        # 양쪽 폴더를 키 기준으로 매핑한 뒤 공통 키만 추출
        visible_map = {
            extract_sample_key(path): path
            for path in sorted(self.visible_dir.iterdir())
            if path.is_file()
        }
        thermal_map = {
            extract_sample_key(path): path
            for path in sorted(self.thermal_dir.iterdir())
            if path.is_file()
        }

        common_keys = sorted(set(visible_map) & set(thermal_map))
        if not common_keys:
            raise FileNotFoundError(
                f"No matched test pairs found in '{self.visible_dir}' and '{self.thermal_dir}'."
            )

        return [
            {
                "key": key,
                "visible_path": visible_map[key],
                "thermal_path": thermal_map[key],
            }
            for key in common_keys
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        # cv2는 기본적으로 BGR로 읽음
        visible = cv2.imread(str(sample["visible_path"]))
        thermal = cv2.imread(str(sample["thermal_path"]))

        if visible is None:
            raise FileNotFoundError(f"Failed to read visible image: {sample['visible_path']}")
        if thermal is None:
            raise FileNotFoundError(f"Failed to read thermal image: {sample['thermal_path']}")

        # 모델은 RGB 입력을 기대하므로 변환
        visible_rgb = cv2.cvtColor(visible, cv2.COLOR_BGR2RGB)
        # albumentations의 additional_targets로 VIS와 Thermal에 동일한 변환 적용
        augmented = self.augmentation(image=visible_rgb, image1=thermal)
        visible_aug = augmented["image"]
        thermal_aug = augmented["image1"]

        # 전처리 (정규화 + HWC → CHW 텐서 변환)
        visible_tensor = self.preprocessing(image=visible_aug)["image"]
        thermal_tensor = self.preprocessing(image=thermal_aug)["image"]

        return {
            "key": sample["key"],
            "visible_tensor": visible_tensor,
            "thermal_tensor": thermal_tensor,
        }


def build_model(encoder: str, encoder_weights: str, device: torch.device) -> torch.nn.Module:
    # 학습 시 구성과 동일하게 모델 생성 (Dual Encoder + Contrastive head)
    # checkpoint의 키와 일치해야 하므로 fusion/contrastive 옵션을 그대로 켜둠
    model = Unet(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        encoder_depth=5,
        classes=1,
        activation="tanh",
        fusion=True,
        contrastive=True,
    )
    model.to(device)
    model.eval()
    return model


def run_inference(args):
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent

    # 데이터셋 루트: --dataset_root 미지정 시 후보 폴더에서 자동 탐색
    dataset_root = (
        Path(args.dataset_root).resolve()
        if args.dataset_root
        else resolve_existing_path(
            project_root / "Dataset" / "dataset_CIDIS_guided_x8x16",
            project_root / "Dataset" / "dataset_CIDIS_guided_x8_x16",
        )
    )

    # 입력 폴더 경로 결정 (CLI 인자가 우선, 없으면 데이터셋 루트 기준 기본 구조 사용)
    if args.visible_dir is None:
        visible_dir = dataset_root / "visible" / "test" / f"guided_{args.scale}"
    else:
        visible_dir = Path(args.visible_dir).resolve()

    if args.thermal_dir is None:
        thermal_dir = dataset_root / "thermal" / "test" / f"guided_{args.scale}" / f"LR_{args.scale}"
    else:
        thermal_dir = Path(args.thermal_dir).resolve()

    # 체크포인트 경로 결정
    checkpoint_path = (
        Path(args.checkpoint).resolve()
        if args.checkpoint
        else resolve_existing_path(
            project_root / "best_model.pth",
            project_root.parent / "best_model.pth",
        )
    )

    # SR 결과 저장 폴더 (기본: outputs/YYYYMMDD-HHMMSS/)
    # 실행 시각으로 하위 폴더를 만들어 이전 결과를 덮어쓰지 않도록 함
    run_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else project_root / "outputs" / run_stamp
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # 필수 경로 검증
    if not visible_dir.exists():
        raise FileNotFoundError(f"Visible test directory not found: {visible_dir}")
    if not thermal_dir.exists():
        raise FileNotFoundError(f"Thermal test directory not found: {thermal_dir}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # 디바이스 자동 선택 (GPU 사용 가능 시 cuda)
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    # 데이터 로더 (추론은 순서 유지를 위해 shuffle=False)
    dataset = ThermalTestDataset(
        visible_dir=visible_dir,
        thermal_dir=thermal_dir,
        encoder=args.encoder,
        encoder_weights=args.encoder_weights,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # 모델 생성 후 체크포인트 가중치 로드 (strict=True: 키 누락/추가 시 에러)
    model = build_model(args.encoder, args.encoder_weights, device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)

    # 추론 루프: 모든 입력 페어에 대해 SR 이미지를 생성하여 저장
    saved_count = 0
    with torch.no_grad():
        for batch in dataloader:
            visible_tensor = batch["visible_tensor"].to(device)
            thermal_tensor = batch["thermal_tensor"].to(device)
            # model.predict는 contrastive head 출력을 제외하고 SR 마스크만 반환
            predictions = model.predict(visible_tensor, thermal_tensor)

            for index in range(predictions.shape[0]):
                key = batch["key"][index]
                prediction_image = tensor_to_uint8_image(predictions[index])
                sr_path = output_dir / f"{key}_sr.png"
                cv2.imwrite(str(sr_path), prediction_image)
                saved_count += 1

    # 실행 결과 요약 출력
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Visible test dir: {visible_dir}")
    print(f"Thermal test dir: {thermal_dir}")
    print(f"SR images saved to: {output_dir}")
    print(f"Processed {saved_count} samples on {device}.")


def build_parser():
    # CLI 인자 정의
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default=None)            # 데이터셋 루트 폴더
    parser.add_argument("--scale", type=str, default="x8", choices=["x8", "x16"])  # 업스케일 배율
    parser.add_argument("--visible_dir", type=str, default=None)             # VIS HR 입력 폴더
    parser.add_argument("--thermal_dir", type=str, default=None)             # Thermal LR 입력 폴더
    parser.add_argument("--checkpoint", type=str, default=None)              # 학습된 모델 가중치 경로
    parser.add_argument("--output_dir", type=str, default=None)              # SR 결과 저장 폴더
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")                # auto / cuda / cpu
    parser.add_argument("--encoder", type=str, default="resnet34")           # 학습 때 사용한 인코더와 동일해야 함
    parser.add_argument("--encoder_weights", type=str, default="imagenet")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
