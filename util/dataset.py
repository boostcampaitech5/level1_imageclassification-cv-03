import os
import random
from collections import defaultdict
from enum import Enum
from typing import Tuple, List

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, Subset, random_split
from torchvision.transforms import Resize, ToTensor, Normalize, Compose, CenterCrop, ColorJitter
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
import torchvision.transforms as T
from sklearn.model_selection import KFold

IMG_EXTENSIONS = [
    ".jpg", ".JPG", ".jpeg", ".JPEG", ".png",
    ".PNG", ".ppm", ".PPM", ".bmp", ".BMP",
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


class MaskLabels(int, Enum):
    MASK = 0
    INCORRECT = 1
    NORMAL = 2


class GenderLabels(int, Enum):
    MALE = 0
    FEMALE = 1

    @classmethod
    def from_str(cls, value: str) -> int:
        value = value.lower()
        if value == "male":
            return cls.MALE
        elif value == "female":
            return cls.FEMALE
        else:
            raise ValueError(f"Gender value should be either 'male' or 'female', {value}")


class AgeLabels(int, Enum):
    YOUNG = 0
    MIDDLE = 1
    OLD = 2

    @classmethod
    def from_number(cls, value: str, age_split) -> int:
        try:
            value = int(value)
        except Exception:
            raise ValueError(f"Age value should be numeric, {value}")

        if value < age_split[0]:
            return cls.YOUNG
        elif value < age_split[1]:
            return cls.MIDDLE
        else:
            return cls.OLD


class Subset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
    
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]
    
    def __len__(self):
        return len(self.indices)
    
    def get_class_weight(self):
        labels = []
        for i in self.indices:
            labels.append(int(self.dataset.get_label(i)[2]))
        labels_cnt = Counter(labels)
        weights = [1/labels_cnt[label] for label in range(len(labels_cnt))]
        weights = [weight / sum(weights) * len(labels_cnt) for weight in weights]
        return weights     
        
    def oversample(self, category='age', weight=[1,1,1]):
        print(f"{category} 기준 오버 샘플링, weight = {weight}")
        category_idx = {'mask':0, 'gender':1, 'age':2}
        labels = []
        for i in self.indices:
            labels.append(int(self.dataset.get_label(i)[category_idx[category]]))
        labels_cnt = Counter(labels)
        print("오버 샘플링 전 클래스 분포 :", labels_cnt)
        
        for key in labels_cnt.keys():
            labels_cnt[key] = int(labels_cnt[key] * weight[key])
        sampler = RandomOverSampler(sampling_strategy=labels_cnt)
        self.indices, labels = sampler.fit_resample(np.array(self.indices).reshape(-1, 1), np.array(labels).reshape(-1, 1))
        self.indices = self.indices.flatten()
        print("오버 샘플링 후 클래스 분포 :", Counter(labels))
    

class MaskBaseDataset(Dataset):
    _file_names = {
        "mask1": MaskLabels.MASK,
        "mask2": MaskLabels.MASK,
        "mask3": MaskLabels.MASK,
        "mask4": MaskLabels.MASK,
        "mask5": MaskLabels.MASK,
        "incorrect_mask": MaskLabels.INCORRECT,
        "normal": MaskLabels.NORMAL
    }

    image_paths = []
    mask_labels = []
    gender_labels = []
    age_labels = []

    def __init__(self, data_dir, age_split, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), val_ratio=0.2):
        self.data_dir = data_dir
        self.mean = mean
        self.std = std
        self.val_ratio = val_ratio
        self.age_split = age_split
        self.num_classes = 3 + 2 + 3
            
        self.transform = None
        self.setup()
        self.calc_statistics()

    def setup(self):
        profiles = os.listdir(self.data_dir)
        for profile in profiles:
            if profile.startswith("."):  # "." 로 시작하는 파일은 무시합니다
                continue

            img_folder = os.path.join(self.data_dir, profile)
            for file_name in os.listdir(img_folder):
                _file_name, ext = os.path.splitext(file_name)
                if _file_name not in self._file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                    continue

                img_path = os.path.join(self.data_dir, profile, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                mask_label = self._file_names[_file_name]

                id, gender, race, age = profile.split("_")
                gender_label = GenderLabels.from_str(gender)
                age_label = AgeLabels.from_number(age, self.age_split)

                self.image_paths.append(img_path)
                self.mask_labels.append(mask_label)
                self.gender_labels.append(gender_label)
                self.age_labels.append(age_label)
        
    def calc_statistics(self):
        has_statistics = self.mean is not None and self.std is not None
        if not has_statistics:
            print("[Warning] Calculating statistics... It can take a long time depending on your CPU machine")
            sums = []
            squared = []
            for image_path in self.image_paths[:3000]:
                image = np.array(Image.open(image_path)).astype(np.int32)
                sums.append(image.mean(axis=(0, 1)))
                squared.append((image ** 2).mean(axis=(0, 1)))

            self.mean = np.mean(sums, axis=0) / 255
            self.std = (np.mean(squared, axis=0) - self.mean ** 2) ** 0.5 / 255

    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, index):
        assert self.transform is not None, ".set_tranform 메소드를 이용하여 transform 을 주입해주세요"
        
        image = self.read_image(index)
        image_transform = self.transform(image)
        
        return image_transform, self.get_label(index)

    def __len__(self):
        return len(self.image_paths)

    def get_label(self, index):
        return self.mask_labels[index], self.gender_labels[index], self.age_labels[index]
    
    def get_labels(self):
        return self.mask_labels, self.gender_labels, self.age_labels
    
    def read_image(self, index):
        image_path = self.image_paths[index]
        return Image.open(image_path)


    @staticmethod
    def denormalize_image(image, mean, std):
        img_cp = image.copy()
        img_cp *= std
        img_cp += mean
        img_cp *= 255.0
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)
        return img_cp

    def split_dataset(self, oversampling=False, category='age', weights=[1,1,1]) -> Tuple[Subset, Subset]:
        """
        데이터셋을 train 과 val 로 나눕니다,
        pytorch 내부의 torch.utils.data.random_split 함수를 사용하여
        torch.utils.data.Subset 클래스 둘로 나눕니다.
        구현이 어렵지 않으니 구글링 혹은 IDE (e.g. pycharm) 의 navigation 기능을 통해 코드를 한 번 읽어보는 것을 추천드립니다^^
        """
        n_val = int(len(self) * self.val_ratio)
        n_train = len(self) - n_val
        train_set, val_set = random_split(self, [n_train, n_val])
        if oversampling:
            train_set.oversample(category, weights)
        return train_set, val_set


class MaskSplitByProfileDataset(MaskBaseDataset):
    """
        train / val 나누는 기준을 이미지에 대해서 random 이 아닌
        사람(profile)을 기준으로 나눕니다.
        구현은 val_ratio 에 맞게 train / val 나누는 것을 이미지 전체가 아닌 사람(profile)에 대해서 진행하여 indexing 을 합니다
        이후 `split_dataset` 에서 index 에 맞게 Subset 으로 dataset 을 분기합니다.
    """

    def __init__(self, data_dir, age_split, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), val_ratio=0.2):
        self.indices = defaultdict(list)
        super().__init__(data_dir, age_split, mean, std, val_ratio)

    @staticmethod
    def _split_profile(profiles, val_ratio):
        length = len(profiles)
        n_val = int(length * val_ratio)

        val_indices = set(random.sample(range(length), k=n_val))
        train_indices = set(range(length)) - val_indices
        return {
            "train": train_indices,
            "val": val_indices
        }     
    
    # def setup(self):
    #     profiles = os.listdir(self.data_dir)
    #     profiles = [profile for profile in profiles if not profile.startswith(".")]
    #     split_profiles = self._split_profile(profiles, self.val_ratio)

    #     cnt = 0
    #     for phase, indices in split_profiles.items():
    #         for _idx in indices:
    #             profile = profiles[_idx]
    #             img_folder = os.path.join(self.data_dir, profile)
    #             for file_name in os.listdir(img_folder):
    #                 _file_name, ext = os.path.splitext(file_name)
    #                 if _file_name not in self._file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
    #                     continue

    #                 img_path = os.path.join(self.data_dir, profile, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
    #                 mask_label = self._file_names[_file_name]

    #                 id, gender, race, age = profile.split("_")
    #                 gender_label = GenderLabels.from_str(gender)
    #                 age_label = AgeLabels.from_number(age, self.age_split)

    #                 self.image_paths.append(img_path)
    #                 self.mask_labels.append(mask_label)
    #                 self.gender_labels.append(gender_label)
    #                 self.age_labels.append(age_label)

    #                 self.indices[phase].append(cnt)
    #                 cnt += 1
    
    def split_dataset(self, oversampling=False, category='age', weights=[1,1,1]) -> List[Subset]:
        result = [Subset(self, indices) for phase, indices in self.indices.items()]
        if oversampling:
            result[0].oversample(category, weights)
        return result 
    
    
    def kfold_split(self, k = 5, oversampling=False, category='age', weights=[1,1,1]) -> List[Subset]:
        profiles = os.listdir(self.data_dir)
        profiles = [profile for profile in profiles if not profile.startswith(".")]
        
        length = len(profiles)
        kf = KFold(n_splits=k, shuffle=True)
        profiles_idx = list(range(0, length*7, 7))
        folds = kf.split(profiles_idx)
        
        result = []
        for train_profile_ids, val_profile_ids in folds:
            train_ids = []
            val_ids = []
            for i in train_profile_ids:
                train_ids.extend(list(range(profiles_idx[i], profiles_idx[i]+7)))
            for i in val_profile_ids:
                val_ids.extend(list(range(profiles_idx[i], profiles_idx[i]+7)))
            result.append([Subset(self, train_ids), Subset(self, val_ids)])
            
        if oversampling:
            for i in range(k):
                print(f"[{i+1}번 Fold]")
                result[i][0].oversample(category, weights)
                print()
        return result 


class TestDataset(Dataset):
    def __init__(self, img_paths, resize, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)):
        self.img_paths = img_paths
        self.transform = Compose([
            Resize(resize, Image.BILINEAR),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)
