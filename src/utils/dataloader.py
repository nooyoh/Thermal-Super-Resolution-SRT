from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import cv2
from .misc import normalize_data, list_img

class Dataset(BaseDataset):
    def __init__(
            self,
            hr_dir: str,
            thermal_dir:str,
            tar_dir: str,
            augmentation=None,
            preprocessing=None,
    ):
        self.hr_list = list_img(hr_dir)
        self.thermal_list = list_img(thermal_dir)
        self.tar_list = list_img(tar_dir)

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        # VIS HR 이미지 로드 (BGR → RGB 변환)
        himage = cv2.imread(self.hr_list[i])
        himage = cv2.cvtColor(himage, cv2.COLOR_BGR2RGB)
        # Thermal HR GT 로드 (그레이스케일)
        target = cv2.imread(self.tar_list[i], 0)
        # Thermal LR 이미지 로드
        timage = cv2.imread(self.thermal_list[i])

        if self.augmentation:
            sample = self.augmentation(image=himage, image1=timage, mask=target)
            himage, target, timage = sample['image'], sample['mask'], sample['image1']

        # target을 (H, W, 1) 형태로 변환하여 채널 차원 추가
        target = target.reshape(480, 640, 1)

        if self.preprocessing:
            sample = self.preprocessing(image=himage, mask=target)
            himage, target = sample['image'], sample['mask']
            sample = self.preprocessing(image=timage)
            timage = sample['image']
            # target을 [0, 1]로 정규화 후 tanh 범위 [-1, 1]로 변환
            target = target / 255
            target = normalize_data(target)

        return himage, timage, target

    def __len__(self):
        return len(self.hr_list)
