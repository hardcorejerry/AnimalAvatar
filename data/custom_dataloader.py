# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

'''
import numpy as np
from torch.utils.data import Dataset
from pytorch3d.renderer.cameras import PerspectiveCameras


class CustomSingleVideo(Dataset):
    """
    An example custom dataset class
    To train Animal-Avatar on a custom scene, complete this Dataset (except get_sparse_keypoints()) with your data
    """

    def __init__(self, *args, **kwargs):
        pass

    def __len__(self):
        raise NotImplementedError

    def get_masks(self, list_items: list[int]) -> np.ndarray:
        """
        Return:
            - masks [len(list_items),H,W,1] NPFLOAT32 [0,1](foreground probability)
        """
        raise NotImplementedError

    def get_imgs_rgb(self, list_items: list[int]) -> np.ndarray:
        """
        Return:
            - imgs_rgb [len(list_items),H,W,3] NPFLOAT32 [0,1] (rgb frames)
        """
        raise NotImplementedError

    def get_cameras(self, list_items: list[int]) -> PerspectiveCameras:
        """
        Return:
            - cameras (len(list_items),) PerspectiveCameras in ndc_space
        """
        raise NotImplementedError

    def get_sparse_keypoints(self, list_items: list[int]) -> tuple[np.ndarray, np.ndarray]:
        """
        Optional!
        Returns:
            - sparse_keypoints (N, N_KPTS, 2) NPFLOAT32
            - scores (N, N_KPTS, 1) NPFLOAT32
        """
        assert False, ("No GT sparse-keypoints are available for custom scenes.", "Please set exp.l_optim_sparse_kp=0 in config/config.yaml", "This will affect performance quality.")
'''

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pytorch3d.renderer.cameras import PerspectiveCameras
from typing import List, Tuple
from PIL import Image
import cv2

class CustomSingleVideo(Dataset):
    """
    完整实现的自定义数据集加载器，兼容AnimalAvatar训练流程
    数据目录结构要求：
    custom_root/
        images/         # RGB图像序列 (0000.png, 0001.png, ...)
        masks/          # 分割掩码序列 (0000.npy, 0001.npy, ...)
        cameras/        # 相机参数文件 (cameras.npz)
    """

    def __init__(self, custom_root: str, img_size: tuple = (800, 800), preload: bool = True):
        """
        Args:
            custom_root: 自定义数据集根目录
            img_size: 输出图像尺寸 (H, W)
            preload: 是否预加载所有数据到内存
        """
        super().__init__()
        self.custom_root = custom_root
        self.img_size = img_size
        self.preload = preload
        
        # 初始化路径
        self.image_dir = os.path.join(custom_root, "images")
        self.mask_dir = os.path.join(custom_root, "masks")
        self.camera_path = os.path.join(custom_root, "cameras/cameras.npz")
        
        # 加载元数据
        self._validate_data_structure()
        self._load_camera_parameters()
        
        # 预加载数据
        if preload:
            self.images = [self._load_image(i) for i in range(len(self))]
            self.masks = [self._load_mask(i) for i in range(len(self))]
        else:
            self.images = None
            self.masks = None

    def _validate_data_structure(self):
        """验证数据目录结构完整性"""
        assert os.path.isdir(self.image_dir), f"Missing images directory: {self.image_dir}"
        assert os.path.isdir(self.mask_dir), f"Missing masks directory: {self.mask_dir}"
        assert os.path.isfile(self.camera_path), f"Missing camera file: {self.camera_path}"
        
        # 检查帧数一致性
        n_images = len(os.listdir(self.image_dir))
        n_masks = len(os.listdir(self.mask_dir))
        assert n_images == n_masks, f"Frame count mismatch: {n_images} images vs {n_masks} masks"

    def _load_camera_parameters(self):
        """加载并转换相机参数到PyTorch3D格式"""
        camera_data = np.load(self.camera_path)
        
        # 转换为NDC空间坐标
        H, W = self.img_size
        scale = 2.0 / min(H, W)
        
        self.cam_params = {
            "focal_length": torch.from_numpy(camera_data["focal"] * scale).float(),
            "principal_point": torch.from_numpy(camera_data["principal"] * scale - 1).float(),
            "R": torch.from_numpy(camera_data["R"]).float(),
            "T": torch.from_numpy(camera_data["T"]).float()
        }
        
        # 验证参数维度
        assert self.cam_params["focal_length"].shape[0] == len(self), "Camera frame count mismatch"

    def __len__(self) -> int:
        return len(os.listdir(self.image_dir))

    def _load_image(self, index: int) -> np.ndarray:
        """加载并预处理单张图像"""
        img_path = os.path.join(self.image_dir, f"{index:04d}.png")
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        
        # 尺寸调整与归一化
        img = cv2.resize(img, self.img_size[::-1])  # OpenCV使用(W,H)顺序
        return img.astype(np.float32) / 255.0

    def _load_mask(self, index: int) -> np.ndarray:
        """加载并预处理单个掩码"""
        mask_path = os.path.join(self.mask_dir, f"{index:04d}.npy")
        mask = np.load(mask_path)
        
        # 尺寸调整与维度扩展
        mask = cv2.resize(mask, self.img_size[::-1], interpolation=cv2.INTER_NEAREST)
        return mask[..., None].astype(np.float32)  # 增加通道维度

    def get_masks(self, list_items: List[int]) -> np.ndarray:
        """返回批处理后的掩码数据 [N,H,W,1]"""
        if self.preload:
            masks = [self.masks[i] for i in list_items]
        else:
            masks = [self._load_mask(i) for i in list_items]
        return np.stack(masks)

    def get_imgs_rgb(self, list_items: List[int]) -> np.ndarray:
        """返回批处理后的图像数据 [N,H,W,3]"""
        if self.preload:
            imgs = [self.images[i] for i in list_items]
        else:
            imgs = [self._load_image(i) for i in list_items]
        return np.stack(imgs)

    def get_cameras(self, list_items: List[int]) -> PerspectiveCameras:
        """构建PyTorch3D相机对象"""
        return PerspectiveCameras(
            focal_length=self.cam_params["focal_length"][list_items],
            principal_point=self.cam_params["principal_point"][list_items],
            R=self.cam_params["R"][list_items],
            T=self.cam_params["T"][list_items],
            in_ndc=True
        )

    def get_sparse_keypoints(self, list_items: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """自定义数据集不提供关键点数据"""
        raise NotImplementedError(
            "Custom dataset does not support sparse keypoints. "
            "Set 'exp.l_optim_sparse_kp=0' in your config."
        )

    # 可选实现方法 -------------------------------------------------
    def get_crop_bbox_list(self, list_items=None) -> List[tuple]:
        """返回每帧的裁剪边界框(x0,y0,w,h)"""
        if list_items is None:
            return [(0, 0, self.img_size[1], self.img_size[0])] * len(self)
        return [(0, 0, self.img_size[1], self.img_size[0])] * len(list_items)

    def get_init_shape(self) -> Tuple[np.ndarray, np.ndarray]:
        """提供初始化形状参数（可选）"""
        return np.zeros(10, dtype=np.float32), np.zeros(4, dtype=np.float32)