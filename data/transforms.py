
#from alphaction.dataset.transforms import video_transforms as T
import torchvision.transforms as T
from dataclasses import dataclass
import numpy as np
import cv2

cv2.setNumThreads(0)


@dataclass
class TransformsCfg:
    MIN_SIZE_TRAIN: int = 256
    MAX_SIZE_TRAIN: int = 464
    MIN_SIZE_TEST: int = 256
    MAX_SIZE_TEST: int = 464
    PIXEL_MEAN = [0.3911 , 0.3359 , 0.3415]
    PIXEL_STD = [0.2841 , 0.2672 ,0.2741]
    TO_BGR: bool = False

    FRAME_NUM: int = 16  # 16
    FRAME_SAMPLE_RATE: int = 4  # 4

    COLOR_JITTER: bool = True
    HUE_JITTER: float = 20.0
    SAT_JITTER: float = 0.1
    VAL_JITTER: float = 0.1


def build_transforms(cfg=TransformsCfg(), is_train=True):
    # build transforms for training of testing
   
    min_size = cfg.MIN_SIZE_TEST
    max_size = cfg.MAX_SIZE_TEST
    color_jitter = False
    flip_prob = 0

    frame_num = cfg.FRAME_NUM
    sample_rate = cfg.FRAME_SAMPLE_RATE

    if color_jitter:
        color_transform = T.ColorJitter(
            cfg.HUE_JITTER, cfg.SAT_JITTER, cfg.VAL_JITTER
        )

    to_bgr = cfg.TO_BGR
    normalize_transform = T.Normalize(
        mean=cfg.PIXEL_MEAN, std=cfg.PIXEL_STD
    )

    transform = T.Compose(
        [
            # T.TemporalCrop(frame_num, sample_rate),
            T.Resize(size = (min_size, max_size)),
            T.RandomCrop(224),
            T.RandomHorizontalFlip(flip_prob),
            T.ToTensor(),
            normalize_transform,
        ]
    )

    return transform
