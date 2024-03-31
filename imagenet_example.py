from datasets import load_dataset
import torch
import copy
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

preprocess= ResNet50_Weights.IMAGENET1K_V2.transforms()

model = resnet50(weights='DEFAULT')   #weights=ResNet50_Weights.IMAGENET1K_V2)
model.cuda()
model.eval()

criterion = torch.nn.CrossEntropyLoss()

orig_image = ds["train"][0]["image"]
pr_im = preprocess(orig_image.transpose(0, -1))
matplotlib.pyplot.imshow(pr_im.transpose(0, -1))
matplotlib.pyplot.savefig("/datasets/tmp/1.png")
print(model(pr_im[None, :].cuda()).argmax())
print(ds["train"][0]["label"])

adv_image = preprocess(torch.Tensor(copy.deepcopy(orig_image).transpose(0, -1)).to("cuda")).to(torch.float)
print(adv_image.dtype)
adv_image.requires_grad = True
# 
# todo go through preprocess
optimizer_adv = torch.optim.Adam([adv_image], lr=1e-4, weight_decay=1e-3)
# todo iterate until class is correct
for i in range(100):
    outputs = model(adv_image[None, :])
    targets = torch.tensor([5]).cuda()
    optimizer_adv.zero_grad()
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer_adv.step()

pred_adv = torch.softmax(model(adv_image[None, :]), 1)
print(pred_adv.argmax().item())
print((pr_im - adv_image.to("cpu")).max())

matplotlib.pyplot.imshow(adv_image.to("cpu").detach().transpose(0, -1))
matplotlib.pyplot.savefig("/datasets/tmp/2.png")
