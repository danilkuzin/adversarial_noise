import copy
import logging
from typing import Optional

from datasets import load_dataset
import matplotlib.pyplot
import torch
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import functional as F, InterpolationMode

import adversarial_noise

import numpy 
import random

import kornia.geometry.transform


def preprocess(
    img,
    crop_size,
    resize_size: int,
    interpolation: str = 'bilinear',
    antialias: Optional[bool] = True,
):
    img = kornia.geometry.transform.resize(img, resize_size, interpolation=interpolation, antialias=antialias)
    img = kornia.geometry.transform.center_crop(img, (crop_size, crop_size))
    img = F.normalize(
            img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    return img


if __name__ == "__main__":
    torch.manual_seed(100)
    numpy.random.seed(100)
    random.seed(100) 

    logging.basicConfig(level=logging.INFO)
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    target_class = 10
    max_iter = 1000

    # load an image from ImageNet
    ds = load_dataset(
        "imagenet-1k", cache_dir="/datasets/imagenet"
    )  # Need to login to huggingface to accept the TnC
    ds = ds.with_format("torch", device="cpu")

    orig_image = ds["train"][0]["image"]

    matplotlib.pyplot.imshow(orig_image)
    matplotlib.pyplot.savefig("/results/original.png")

    adv_image = adversarial_noise.generate_adversarial_image(
        original_image=orig_image,
        model=model,
        target_class=target_class,
        preprocess=lambda x: preprocess(x, crop_size=224, resize_size=232),
        max_iter=max_iter,
    )

    if adv_image is not None:
        print(
            "original class: ",
            model(
                preprocess(
                    F.convert_image_dtype(
                        orig_image.transpose(0, -1).cuda(), torch.float)[None, :], crop_size=224, resize_size=232
                )
            )
            .argmax()
            .item(),
        )
        print(
            "adversarial class: ",
            model(
                preprocess(
                    adv_image.transpose(0, -1).cuda()[None, :], crop_size=224, resize_size=232
                )
            )
            .argmax()
            .item(),
        )

        matplotlib.pyplot.imshow(adv_image)
        matplotlib.pyplot.savefig("/results/adversarial.png")
