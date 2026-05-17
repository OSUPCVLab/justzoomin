import torch
from PIL import Image
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class AlbumentationsWrapper:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img: Image.Image) -> torch.Tensor:
        img_np = np.array(img)
        return self.transform(image=img_np)["image"]


class Cut(ImageOnlyTransform):
    def __init__(self, cutting=None, always_apply=False, p=1.0):
        super(Cut, self).__init__(always_apply, p)
        self.cutting = cutting

    def apply(self, image, **params):
        if self.cutting and self.cutting > 0:
            image = image[self.cutting:-self.cutting, :, :]
        return image

    def get_transform_init_args_names(self):
        return ("cutting",)


class GroundAugmentThenSplit:
    """
    Apply global augmentations to the full wide ground image FIRST (NumPy),
    keeping 2:1 aspect, then split into left/right square halves and
    convert each half to a normalized tensor. Returns (2, C, H, W).
    """
    def __init__(self, target_size: int, aug_cfg=None):
        self.target = target_size
        self.aug_enabled = (aug_cfg is not None) and bool(aug_cfg.enable)

        if self.aug_enabled:
            scale_lo, scale_hi = aug_cfg.ground_scale
            b, c, s, h = aug_cfg.color_jitter
            gray_p = aug_cfg.gray_p
            blur_p = aug_cfg.blur_p
            erase_p = aug_cfg.erase_p
            cutting = aug_cfg.ground_cutting

            # 2:1 global pipeline (NumPy domain)
            self.global_aug = A.Compose([
                Cut(cutting=cutting, p=1.0),
                A.ImageCompression(quality_range=(90, 100), p=0.5),
                A.RandomResizedCrop(
                    size=(self.target, 2 * self.target),
                    scale=(scale_lo, scale_hi),
                    ratio=(2.0, 2.0),  # keep 2:1
                    interpolation=cv2.INTER_LINEAR_EXACT,
                    p=1.0
                ),
                A.ColorJitter(brightness=b, contrast=c, saturation=s, hue=h, p=0.5),
                A.ToGray(p=gray_p),
                A.OneOf([
                    A.AdvancedBlur(p=1.0),
                    A.Sharpen(p=1.0),
                ], p=blur_p),
            ])

            # Per-half (1:1), then normalize + tensor
            self.per_half_aug = A.Compose([
                A.OneOf([
                    A.GridDropout(ratio=0.5, p=1.0),
                    A.CoarseDropout(
                        num_holes_range=(10, 25),
                        hole_height_range=(int(0.1 * self.target), int(0.2 * self.target)),
                        hole_width_range=(int(0.1 * self.target), int(0.2 * self.target)),
                        fill=0,
                        p=1.0
                    ),
                ], p=erase_p),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2()
            ])
        else:
            # Deterministic resize (validation/test)
            self.global_aug = A.Compose([
                Cut(cutting=0, p=1.0),
                A.Resize(
                    height=self.target,
                    width=2 * self.target,
                    interpolation=cv2.INTER_LINEAR_EXACT,
                    p=1.0
                ),
            ])
            self.per_half_aug = A.Compose([
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2()
            ])

    def __call__(self, img: Image.Image) -> torch.Tensor:
        img_np = np.array(img)
        wide_np = self.global_aug(image=img_np)['image']  # (H, 2H, C)

        # Split into left/right (H, H, C)
        left_np = wide_np[:, :self.target, :]
        right_np = wide_np[:, self.target:, :]

        tl = self.per_half_aug(image=left_np)["image"]
        tr = self.per_half_aug(image=right_np)["image"]

        return torch.stack([tl, tr], dim=0)


def build_satellite_transform(target_image_size, aug_cfg=None):
    """
    Satellite transform.
    If aug_cfg and aug_cfg.enable are set -> apply augs.
    Otherwise -> plain resize + normalize.
    """
    is_train = (aug_cfg is not None) and bool(aug_cfg.enable)

    if is_train:
        b, c, s, h = aug_cfg.color_jitter
        blur_p = aug_cfg.blur_p
        dropout_p = aug_cfg.erase_p  # use same prob for occlusions

        sat_augs = A.Compose([
            A.ImageCompression(quality_range=(90, 100), p=0.5),
            A.Resize(height=target_image_size, width=target_image_size,
                     interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
            A.ColorJitter(brightness=b, contrast=c, saturation=s, hue=h, p=0.5),
            A.OneOf([
                A.AdvancedBlur(p=1.0),
                A.Sharpen(p=1.0),
            ], p=blur_p),
            A.OneOf([
                A.GridDropout(ratio=0.4, p=1.0),
                A.CoarseDropout(
                    num_holes_range=(10, 25),
                    hole_height_range=(int(0.1 * target_image_size), int(0.2 * target_image_size)),
                    hole_width_range=(int(0.1 * target_image_size), int(0.2 * target_image_size)),
                    fill=0,
                    p=1.0
                ),
            ], p=dropout_p),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
        return AlbumentationsWrapper(sat_augs)
    else:
        sat_augs = A.Compose([
            A.Resize(height=target_image_size, width=target_image_size,
                     interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
        return AlbumentationsWrapper(sat_augs)


def build_transforms(target_image_size, aug_cfg=None):
    """
    Returns:
      {
        "ground": GroundAugmentThenSplit(...),
        "satellite": (augmented if aug_cfg.enable),
        "satellite_no_aug": (always plain resize+normalize)
      }
    """
    ground_transform = GroundAugmentThenSplit(target_image_size, aug_cfg=aug_cfg)
    satellite_transform = build_satellite_transform(target_image_size, aug_cfg=aug_cfg)
    satellite_transform_no_aug = build_satellite_transform(target_image_size, aug_cfg=None)

    return {
        "ground": ground_transform,
        "satellite": satellite_transform,
        "satellite_no_aug": satellite_transform_no_aug
    }
