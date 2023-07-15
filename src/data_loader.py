import glob
import os
import pickle
from pathlib import Path
import numpy as np

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from infogan.src.jp_char_list import CHAR_LIST_FOR_SYUKU

class EtlCdbDataLoader(Dataset):
    IMG_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]

    def __init__(self, img_dir, transform=None):
        self.etl_paths, self.np_labels = self._get_etl_paths(img_dir)
        self.transform = transform

    def __getitem__(self, index):
        etl_path = self.etl_paths[index]
        img = Image.open(etl_path[0])
        label = np.where(self.np_labels == etl_path[2])[0][0]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def _get_etl_paths(self, img_dir):
        img_dir = Path(img_dir)
        pickle_paths = glob.glob(os.path.join(img_dir, '*.pickle'))
        etl_paths = []
        temp_labels = []
        for pickle_path in pickle_paths:
            with open(pickle_path, 'rb') as pickle_file:
                contents = pickle.load(pickle_file)
                for content in contents:
                    root_dir = Path(content[0]).parents[3]
                    child_dir = content[0].split(str(root_dir))[1]
                    dir = os.path.join(str(img_dir) + str(child_dir))
                    result = False
                    for target in CHAR_LIST_FOR_SYUKU:
                        if target in dir:
                            result = True
                    if result == True:
                        etl_paths.append([dir, content[1], ord(content[2])])
                        temp_labels.append(ord(content[2]))

        etl_paths = [
            p for p in etl_paths
            if '.' + os.path.splitext(os.path.basename(p[0]))[1][1:] in EtlCdbDataLoader.IMG_EXTENSIONS
        ]

        np_labels = np.sort(np.unique(np.array(temp_labels)))
        return etl_paths, np_labels

    def __len__(self):
        return len(self.etl_paths)

def get_dataloader(opt):
    transform = transforms.Compose([
        transforms.Resize(opt.img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    dataloader = DataLoader(
        EtlCdbDataLoader(
            "ocr_dataset_create_jp/output",
            transform=transform
        ),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu
    )

    return dataloader
