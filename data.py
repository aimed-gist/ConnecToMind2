import os
import re
import glob
import random
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data import ConcatDataset
from torch import nn
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
from torchvision import transforms

from PIL import Image
import nibabel as nib

import utils

class TrainDataset_ourdata(Dataset): # ses단위로 실행
    def __init__(self, fmri_path, stimuli_path, image_path, transform):
        # .npy 파일은 mmap 지원 → 메모리에 전체 로드 없이 필요한 샘플만 읽음
        self.fmri = np.load(fmri_path, mmap_mode='r')  # [N, 100, 3291]
        # stimuli: 이미지 파일명 배열 (예: ['coco2017_14.jpg', 'coco2017_28.jpg', ...])
        self.stimuli = np.load(stimuli_path, allow_pickle=True)  # [N,]
        self.image_path = image_path
        self.transform = transform # PIL.Image -> tensor

        # 파일 경로에서 subject ID 자동 추출 (예: /path/to/sub-01/... -> "sub-01")
        match = re.search(r'(sub-\d+)', fmri_path)
        self.subject_name = match.group(1) if match else "unknown"

    def __len__(self):
        return len(self.stimuli)

    def __getitem__(self, idx):
        # fMRI 데이터 로딩: [100, 3291]
        fmri_vol = torch.tensor(self.fmri[idx], dtype=torch.float32)

        # 이미지 로딩
        img_path = os.path.join(self.image_path, self.stimuli[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return fmri_vol, image, self.subject_name

class TestDataset_ourdata(Dataset): # ses단위로 실행
    def __init__(self, fmri_path, stimuli_path, image_path, transform):
        # .npy 파일은 mmap 지원 → 메모리에 전체 로드 없이 필요한 샘플만 읽음
        self.fmri = np.load(fmri_path, mmap_mode='r')  # [N, 100, 3291]
        # stimuli: 이미지 파일명 배열 (예: ['coco2017_14.jpg', 'coco2017_28.jpg', ...])
        self.stimuli = np.load(stimuli_path, allow_pickle=True)  # [N,]
        self.image_path = image_path
        self.transform = transform # PIL.Image -> tensor

        # 파일 경로에서 subject ID 자동 추출 (예: /path/to/sub-01/... -> "sub-01")
        match = re.search(r'(sub-\d+)', fmri_path)
        self.subject_name = match.group(1) if match else "unknown"

    def __len__(self):
        return len(self.stimuli)

    def __getitem__(self, idx):
        # fMRI 데이터 로딩: [100, 3291]
        fmri_vol = torch.tensor(self.fmri[idx], dtype=torch.float32)

        # 이미지 로딩
        img_path = os.path.join(self.image_path, self.stimuli[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return fmri_vol, image, self.stimuli[idx], self.subject_name
    

def train_dataset(args):
    root_dir = args.root_dir
    fmri_dir = args.fmri_dir
    fmri_detail_dir = args.fmri_detail_dir
    image_dir = args.image_dir
    subjects = args.subjects
    roi_suffix = args.roi_suffix  # e.g., 'dk', 'destrieux', 'schaefer', etc.
    transform = transforms.ToTensor()

    datasets = []
    for sub in subjects:
        #fmri_path = f"{root_dir}/{fmri_dir}/{fmri_detail_dir}/{sub}/{sub}_beta-train_{roi_suffix}.npy"
        fmri_path = f"{root_dir}/{fmri_dir}/{sub}/{sub}_beta-train_{roi_suffix}.npy"
        #stimuli_path = f"{root_dir}/{fmri_dir}/{fmri_detail_dir}/{sub}/{sub}_stimuli-train.npy"
        stimuli_path = f"{root_dir}/{fmri_dir}/{sub}/{sub}_stimuli-train.npy"
        image_path = f"{root_dir}/{image_dir}"
        datasets.append(TrainDataset_ourdata(fmri_path, stimuli_path, image_path, transform))

    train_dataset = ConcatDataset(datasets)
    return train_dataset

def test_dataset(args):
    """Subject별로 분리된 test dataset dictionary 반환"""
    root_dir = args.root_dir
    fmri_dir = args.fmri_dir
    fmri_detail_dir = args.fmri_detail_dir
    image_dir = args.image_dir
    subjects = args.subjects
    roi_suffix = args.roi_suffix  # e.g., 'dk', 'destrieux', 'schaefer', etc.
    transform = transforms.ToTensor()

    datasets = {}
    for sub in subjects:
        #fmri_path = f"{root_dir}/{fmri_dir}/{fmri_detail_dir}/{sub}/{sub}_beta-test_{roi_suffix}.npy"
        fmri_path = f"{root_dir}/{fmri_dir}/{sub}/{sub}_beta-test_{roi_suffix}.npy"
        #stimuli_path = f"{root_dir}/{fmri_dir}/{fmri_detail_dir}/{sub}/{sub}_stimuli-test.npy"
        stimuli_path = f"{root_dir}/{fmri_dir}/{sub}/{sub}_stimuli-test.npy"
        image_path = f"{root_dir}/{image_dir}"
        datasets[sub] = TestDataset_ourdata(fmri_path, stimuli_path, image_path, transform)

    return datasets

def get_dataloader(args):
    '''
        train_loader.shape
            fmri_vol.shape  [batch_size, roir개수, (voxel개수+padding)],
            image.shape  [batch_size, 3, 224, 224]
        test_loaders: dict[subject, DataLoader]
            fmri_vol.shape  [inference_batch_size, roir개수, (voxel개수+padding)],
            image.shape  [inference_batch_size, 3, 224, 224]
            image_id.shape  [inference_batch_size,]  ex) ['coco2017_14.jpg' 'coco2017_14.jpg', ...]
    '''
    # 제거할 index 집합
    # drop_idx = {0, 5, 8, 10, 15, 18} # low
    # drop_idx = {1,4,11,14} # high

    if args.mode == 'train':
        full_ds = train_dataset(args)

        # Train/Val split (seed 고정으로 매 실행마다 동일한 분리)
        val_size = args.val_size  # default: 10000
        train_size = len(full_ds) - val_size
        train_ds, val_ds = random_split(
            full_ds,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(args.seed)
        )

        # train: drop_last=True - all_gather 시 배치 크기 일치 보장 + 중복 샘플 방지 (BLIP-2 공식 방식)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=args.num_workers, prefetch_factor=args.prefetch_factor, persistent_workers=True, pin_memory=True, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.num_workers, prefetch_factor=args.prefetch_factor, persistent_workers=True, pin_memory=True, shuffle=False, drop_last=False)
        return train_loader, val_loader

    if args.mode == 'inference':
        test_datasets = test_dataset(args)
        test_loaders = {}
        for sub, ds in test_datasets.items():
            test_loaders[sub] = DataLoader(ds, batch_size=args.inference_batch_size, num_workers=args.num_workers, prefetch_factor=args.prefetch_factor, persistent_workers=True, pin_memory=True, shuffle=False)
        return test_loaders
    
def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32  # worker별 고유 seed 생성
    np.random.seed(seed)
    random.seed(seed)    

def collate_fn_factory_train(keep_idx):
    """
    keep_idx 리스트를 클로저로 잡는 collate_fn 생성기
    """
    def collate_fn(batch):
        # batch: list of tuples [(fmri, label), ...]
        fmri_batch, label_batch = zip(*batch)
        fmri_batch = torch.stack(fmri_batch, dim=0)        # [B, 20, 2056]
        fmri_batch = fmri_batch[:, keep_idx, :]            # [B, len(keep_idx), 2056]
        label_batch = torch.stack(label_batch, dim=0)
        return fmri_batch, label_batch
    return collate_fn

def collate_fn_factory_test(keep_idx):
    """
    keep_idx 리스트를 클로저로 잡는 collate_fn 생성기
    """
    def collate_fn(batch):
        # batch: list of tuples [(fmri_vol, image, low_image, image_id), ...]
        fmri_list, image_list, low_list, id_list = zip(*batch)

        # 1) fmri: [B, seq_len, feats]
        fmri_batch = torch.stack(fmri_list, dim=0)
        # 필요한 ROI만 남기기 (keep_idx 는 미리 정의된 리스트)
        fmri_batch = fmri_batch[:, keep_idx, :]

        # 2) image: [B, C, H, W] (transform이 Tensor 변환까지 했을 경우)
        image_batch = torch.stack(image_list, dim=0)

        # 3) low_image: use_low_image 여부에 따라 텐서 혹은 빈 리스트
        if isinstance(low_list[0], torch.Tensor):
            low_batch = torch.stack(low_list, dim=0)
        else:
            # low_image가 [] 로 들어오는 경우, 그냥 빈 리스트 묶음으로 전달
            low_batch = list(low_list)

        # 4) image_id: 문자열 ID 리스트
        id_batch = list(id_list)

        # 최종 반환: fmri, image, low_image, id
        return fmri_batch, image_batch, low_batch, id_batch
    return collate_fn