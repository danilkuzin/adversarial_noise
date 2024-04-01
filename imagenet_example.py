from datasets import load_dataset
import torch
from torch import Tensor
from torchvision.transforms import functional as F, InterpolationMode
import copy
from typing import Optional, Tuple

# import torchvision
from torchvision.models import resnet50, ResNet50_Weights
import matplotlib.pyplot

batch_size = 1
workers = 0

ds = load_dataset(
    "imagenet-1k", cache_dir="/datasets/imagenet"
)  # Need to login to huggingface to accept the TnC
ds = ds.with_format("torch", device="cpu")

train_loader = torch.utils.data.DataLoader(
    ds["train"],
    batch_size=batch_size,
    shuffle=True,
    num_workers=workers,
    pin_memory=True,
)

val_loader = torch.utils.data.DataLoader(
    ds["validation"],
    batch_size=batch_size,
    shuffle=False,
    num_workers=workers,
    pin_memory=True,
)

# print(ds["train"][0])


# preprocess= ResNet50_Weights.IMAGENET1K_V2.transforms()
def preprocess_int(
    img,
    crop_size,
    resize_size: int,
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    antialias: Optional[bool] = True,
):
    img = F.resize(img, resize_size, interpolation=interpolation, antialias=antialias)
    img = F.center_crop(img, crop_size)

    return img


def preprocess(
    img: Tensor,
    crop_size=224,
    resize_size: int = 256,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    antialias: Optional[bool] = True,
) -> Tensor:
    img = preprocess_int(
        img, crop_size, resize_size, interpolation, antialias
    )

    img = F.convert_image_dtype(img, torch.float)
    img = F.normalize(img, mean=mean, std=std)
    return img


def reverse_image_int(
    original_image,
    adv_image_float,
    resize_size = 256, 
    crop_size = 224,
    interpolation = InterpolationMode.BILINEAR, 
    antialias = True
):
    if isinstance(crop_size, int):
        crop_size = (int(crop_size), int(crop_size))
    original_image_dtype = original_image.dtype
    _, original_image_height, original_image_width = F.get_dimensions(original_image)
    resized_image = F.resize(original_image, resize_size, interpolation=interpolation, antialias=antialias)
    
    crop_height, crop_width = crop_size
    _, image_height, image_width = F.get_dimensions(resized_image)

    crop_top = int(round((image_height - crop_height) / 2.0))
    crop_left = int(round((image_width - crop_width) / 2.0))

    resized_image[:, crop_top:crop_top+crop_height, crop_left:crop_left+crop_width] = (adv_image_float*255).to(torch.uint8)

    return F.resize(resized_image, [original_image_height, original_image_width])
    


model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
model.cuda()
model.eval()

criterion = torch.nn.CrossEntropyLoss()

orig_image = ds["train"][0]["image"]
matplotlib.pyplot.imshow(orig_image)
matplotlib.pyplot.savefig("/datasets/tmp/0.png")
copy_orig_image = copy.deepcopy(orig_image)
pr_im = ResNet50_Weights.IMAGENET1K_V2.transforms()(copy_orig_image.transpose(0, -1))
matplotlib.pyplot.imshow(pr_im.transpose(0, -1))
matplotlib.pyplot.savefig("/datasets/tmp/1.png")
print(model(pr_im[None, :].cuda()).argmax())
print(ds["train"][0]["label"])

adv_image_int = preprocess_int(copy_orig_image.transpose(0, -1).cuda(), crop_size=224, resize_size=232)
# adv_image_int = F.pil_to_tensor(adv_image_int)
adv_image_float = F.convert_image_dtype(adv_image_int, torch.float)
adv_image_float.requires_grad = True
adv_image_float_transformed = F.normalize(adv_image_float, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
adv_image_float_transformed = adv_image_float_transformed[None, :]

print(model(adv_image_float_transformed).argmax().item())

# adv_image = copy.deepcopy(orig_image).to("cuda")
# adv_image = preprocess(adv_image.transpose(0, -1)).to(torch.float)
# adv_image.requires_grad = True
# print(model(adv_image[None, :]).argmax())

# print(adv_image.dtype)

# adv_image = preprocess(torch.Tensor(copy.deepcopy(orig_image).transpose(0, -1)).to("cuda")).to(torch.float)
# print(adv_image.dtype)
# adv_image.requires_grad = True
# print(model(adv_image[None, :]).argmax())

optimizer_adv = torch.optim.Adam([adv_image_float], lr=1e-4, weight_decay=1e-3)
# todo iterate until class is correct
for i in range(100):
    # outputs = model(adv_image_float_transformed)
    outputs = model(
        F.normalize(adv_image_float, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))[None, :]
    )
    targets = torch.tensor([54]).cuda()
    optimizer_adv.zero_grad()
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer_adv.step()

pred_adv = torch.softmax(model(
        F.normalize(adv_image_float, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))[None, :]
    ), 1)
print(pred_adv.argmax().item())
# print((orig_image - adv_image_float.transpose(0, -1).to("cpu")).max())

print(adv_image_float.max())
print(adv_image_float.min())



matplotlib.pyplot.imshow((adv_image_float*255).transpose(0, -1).to("cpu").to(torch.uint8).detach())
matplotlib.pyplot.savefig("/datasets/tmp/2.png")

full_adv_image = reverse_image_int(orig_image.transpose(0, -1), adv_image_float.to("cpu").detach(), resize_size=232)

matplotlib.pyplot.imshow(full_adv_image.transpose(0, -1))
matplotlib.pyplot.savefig("/datasets/tmp/3.png")
