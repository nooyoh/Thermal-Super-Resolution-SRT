import argparse
from pathlib import Path
from typing import List, Dict

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
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def extract_sample_key(path: Path) -> str:
    stem = path.stem
    for suffix in ("_vis", "_th"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def tensor_to_uint8_image(tensor: torch.Tensor) -> np.ndarray:
    image = tensor.detach().cpu().squeeze().numpy()
    image = np.clip((image + 1.0) / 2.0, 0.0, 1.0)
    image = (image * 255.0).round().astype(np.uint8)
    return image


def compute_psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    mse = np.mean((pred.astype(float) - gt.astype(float)) ** 2)
    return 10 * np.log10(255 ** 2 / mse) if mse > 0 else 100.0


def compute_ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(float)
    gt = gt.astype(float)
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    mu_pred = np.mean(pred)
    mu_gt = np.mean(gt)
    sigma_pred = np.var(pred)
    sigma_gt = np.var(gt)
    sigma_cross = np.mean((pred - mu_pred) * (gt - mu_gt))
    numerator = (2 * mu_pred * mu_gt + C1) * (2 * sigma_cross + C2)
    denominator = (mu_pred ** 2 + mu_gt ** 2 + C1) * (sigma_pred + sigma_gt + C2)
    return float(numerator / denominator)


def build_gt_map(gt_dir: Path) -> Dict[str, Path]:
    # GT 폴더에서 key → 파일 경로 매핑 생성
    return {
        extract_sample_key(path): path
        for path in sorted(gt_dir.iterdir())
        if path.is_file()
    }


def save_eval_results(
    results: List[Dict],
    eval_metric: str,
    top_k: int,
    eval_dir: Path,
) -> None:
    # eval_metric 기준으로 내림차순 정렬
    sorted_results = sorted(results, key=lambda x: x[eval_metric], reverse=True)
    top_results = sorted_results[:top_k]
    bottom_results = sorted_results[-top_k:]

    top_dir = eval_dir / f"top{top_k}"
    bottom_dir = eval_dir / f"bottom{top_k}"
    top_dir.mkdir(parents=True, exist_ok=True)
    bottom_dir.mkdir(parents=True, exist_ok=True)

    for result in top_results:
        key = result["key"]
        cv2.imwrite(str(top_dir / f"{key}_sr.png"), result["sr_image"])
        if result["preview_image"] is not None:
            cv2.imwrite(str(top_dir / f"{key}_preview.png"), result["preview_image"])

    for result in bottom_results:
        key = result["key"]
        cv2.imwrite(str(bottom_dir / f"{key}_sr.png"), result["sr_image"])
        if result["preview_image"] is not None:
            cv2.imwrite(str(bottom_dir / f"{key}_preview.png"), result["preview_image"])

    print(f"Saved top{top_k} → {top_dir}")
    print(f"Saved bottom{top_k} → {bottom_dir}")


def make_preview_image(
    visible_preview: np.ndarray,
    thermal_preview: np.ndarray,
    prediction_image: np.ndarray,
) -> np.ndarray:
    if thermal_preview.ndim == 3:
        thermal_preview_gray = cv2.cvtColor(thermal_preview, cv2.COLOR_BGR2GRAY)
    else:
        thermal_preview_gray = thermal_preview

    thermal_preview_bgr = cv2.cvtColor(thermal_preview_gray, cv2.COLOR_GRAY2BGR)
    sr_preview_bgr = cv2.cvtColor(prediction_image, cv2.COLOR_GRAY2BGR)
    visible_preview_bgr = cv2.cvtColor(visible_preview, cv2.COLOR_RGB2BGR)

    return np.concatenate([visible_preview_bgr, thermal_preview_bgr, sr_preview_bgr], axis=1)


class ThermalTestDataset(Dataset):
    def __init__(self, visible_dir: Path, thermal_dir: Path, encoder: str, encoder_weights: str):
        self.visible_dir = visible_dir
        self.thermal_dir = thermal_dir
        self.samples = self._build_samples()
        preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weights)
        self.augmentation = get_validation_augmentation()
        self.preprocessing = get_preprocessing(preprocessing_fn)

    def _build_samples(self):
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

        visible = cv2.imread(str(sample["visible_path"]))
        thermal = cv2.imread(str(sample["thermal_path"]))

        if visible is None:
            raise FileNotFoundError(f"Failed to read visible image: {sample['visible_path']}")
        if thermal is None:
            raise FileNotFoundError(f"Failed to read thermal image: {sample['thermal_path']}")

        visible_rgb = cv2.cvtColor(visible, cv2.COLOR_BGR2RGB)
        augmented = self.augmentation(image=visible_rgb, image1=thermal)
        visible_aug = augmented["image"]
        thermal_aug = augmented["image1"]

        visible_tensor = self.preprocessing(image=visible_aug)["image"]
        thermal_tensor = self.preprocessing(image=thermal_aug)["image"]

        return {
            "key": sample["key"],
            "visible_tensor": visible_tensor,
            "thermal_tensor": thermal_tensor,
            "visible_preview": visible_aug,
            "thermal_preview": thermal_aug,
        }


def build_model(encoder: str, encoder_weights: str, device: torch.device) -> torch.nn.Module:
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

    dataset_root = (
        Path(args.dataset_root).resolve()
        if args.dataset_root
        else resolve_existing_path(
            project_root / "Dataset" / "dataset_CIDIS_guided_x8x16",
            project_root / "Dataset" / "dataset_CIDIS_guided_x8_x16",
        )
    )

    if args.visible_dir is None:
        visible_dir = dataset_root / "visible" / "test" / f"guided_{args.scale}"
    else:
        visible_dir = Path(args.visible_dir).resolve()

    if args.thermal_dir is None:
        thermal_dir = dataset_root / "thermal" / "test" / f"guided_{args.scale}" / f"LR_{args.scale}"
    else:
        thermal_dir = Path(args.thermal_dir).resolve()

    checkpoint_path = (
        Path(args.checkpoint).resolve()
        if args.checkpoint
        else resolve_existing_path(
            project_root / "best_model.pth",
            project_root.parent / "best_model.pth",
        )
    )

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else project_root / "outputs" / f"test_{args.scale}"
    )
    sr_dir = output_dir / "sr"
    preview_dir = output_dir / "preview"
    sr_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    if not visible_dir.exists():
        raise FileNotFoundError(f"Visible test directory not found: {visible_dir}")
    if not thermal_dir.exists():
        raise FileNotFoundError(f"Thermal test directory not found: {thermal_dir}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # 평가 모드 설정
    eval_mode = args.gt_dir is not None
    gt_map: Dict[str, Path] = {}
    if eval_mode:
        gt_dir = Path(args.gt_dir).resolve()
        if not gt_dir.exists():
            raise FileNotFoundError(f"GT directory not found: {gt_dir}")
        gt_map = build_gt_map(gt_dir)
        eval_dir = current_file.parent / "eval_results"

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

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

    model = build_model(args.encoder, args.encoder_weights, device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)

    saved_count = 0
    eval_results: List[Dict] = []

    with torch.no_grad():
        for batch in dataloader:
            visible_tensor = batch["visible_tensor"].to(device)
            thermal_tensor = batch["thermal_tensor"].to(device)
            predictions = model.predict(visible_tensor, thermal_tensor)

            for index in range(predictions.shape[0]):
                key = batch["key"][index]
                prediction_image = tensor_to_uint8_image(predictions[index])

                preview_image = None
                if args.save_preview or eval_mode:
                    visible_preview = batch["visible_preview"][index].numpy().astype(np.uint8)
                    thermal_preview = batch["thermal_preview"][index].numpy().astype(np.uint8)
                    preview_image = make_preview_image(visible_preview, thermal_preview, prediction_image)

                if eval_mode:
                    # GT와 비교하여 PSNR/SSIM 계산 후 메모리에 보관
                    if key not in gt_map:
                        continue
                    gt_image = cv2.imread(str(gt_map[key]), cv2.IMREAD_GRAYSCALE)
                    if gt_image is None:
                        continue
                    psnr_value = compute_psnr(prediction_image, gt_image)
                    ssim_value = compute_ssim(prediction_image, gt_image)
                    eval_results.append({
                        "key": key,
                        "psnr": psnr_value,
                        "ssim": ssim_value,
                        "sr_image": prediction_image,
                        "preview_image": preview_image,
                    })
                else:
                    # 기존 동작: 전체 이미지 저장
                    sr_path = sr_dir / f"{key}_sr.png"
                    cv2.imwrite(str(sr_path), prediction_image)

                    if args.save_preview and preview_image is not None:
                        preview_path = preview_dir / f"{key}_preview.png"
                        cv2.imwrite(str(preview_path), preview_image)

                saved_count += 1

    if eval_mode:
        save_eval_results(eval_results, args.eval_metric, args.top_k, eval_dir)
    else:
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Visible test dir: {visible_dir}")
        print(f"Thermal test dir: {thermal_dir}")
        print(f"SR images saved to: {sr_dir}")
        if args.save_preview:
            print(f"Preview images saved to: {preview_dir}")

    print(f"Processed {saved_count} samples on {device}.")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default=None)
    parser.add_argument("--scale", type=str, default="x8", choices=["x8", "x16"])
    parser.add_argument("--visible_dir", type=str, default=None)
    parser.add_argument("--thermal_dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--encoder", type=str, default="resnet34")
    parser.add_argument("--encoder_weights", type=str, default="imagenet")
    parser.add_argument("--save_preview", action="store_true")
    # 평가 모드 인자
    parser.add_argument("--gt_dir", type=str, default=None)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--eval_metric", type=str, default="psnr", choices=["psnr", "ssim"])
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
