from torch.utils.data import Dataset
import pathlib
from PIL import Image
import os
from typing import Tuple, Dict, List
import torch

# class INIT_Dataset(Dataset):
#     def __init__(self, data_cfg, train_mode):
#         self.data_cfg = data_cfg
#         self.train_mode = train_mode
# 
#     def __len__(self):
#         return len(self.data_cfg)
# 
#     def __getitem__(self, index):
#         sample = self.data_cfg[index]
#         if sample is None:
#             print(f"None sample at index {index}")
#         return sample
# 
# class Img2IR_Dataset(Dataset):
#     def __init__(self, data_cfg, train_mode):
#         self.data_cfg = data_cfg
#         self.train_mode = train_mode
# 
#     def __len__(self):
#         return len(self.data_cfg)
# 
#     def __getitem__(self, index):
#         sample = self.data_cfg[index]
#         if sample is None:
#             print(f"None sample at index {index}")
#         return sample



#
def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {directory}.")
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


import torch
from torch.utils.data import Dataset
from PIL import Image
import pathlib
from typing import Tuple


class INIT_Dataset(Dataset):
    def __init__(self, data_cfg,train_mode) -> None:
        self.paths = self.get_image_paths(data_cfg)
        self.classes, self.class_to_idx = self.find_classes(data_cfg)

    def get_image_paths(self, data_cfg: str):
        return list(pathlib.Path(data_cfg).rglob("*.jpg"))

    def find_classes(self, directory: str):
        classes = [d.name for d in pathlib.Path(directory).iterdir() if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def load_image(self, index: int) -> Image.Image:
        "Opens an image via a path and returns it."
        image_path = self.paths[index]
        return Image.open(image_path)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        img = self.load_image(index)
        class_name = self.paths[index].parent.name
        class_idx = self.class_to_idx[class_name]
        return img, class_idx  # return data, label (X, y)

class Img2IR_Dataset(Dataset):
    def __init__(self, data_cfg: str) -> None:
        self.paths = self.get_image_paths(data_cfg)
        self.classes, self.class_to_idx = self.find_classes(data_cfg)

    def get_image_paths(self, data_cfg: str):
        return list(pathlib.Path(data_cfg).rglob("*.jpg"))

    def find_classes(self, directory: str):
        classes = [d.name for d in pathlib.Path(directory).iterdir() if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def load_image(self, index: int) -> Image.Image:
        "Opens an image via a path and returns it."
        image_path = self.paths[index]
        return Image.open(image_path)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        img = self.load_image(index)
        class_name = self.paths[index].parent.name
        class_idx = self.class_to_idx[class_name]
        return img, class_idx  # return data, label (X, y)
