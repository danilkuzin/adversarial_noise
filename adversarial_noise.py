import copy
import logging
import typing

import torch
from torchvision.transforms import functional as F


def generate_adversarial_image(
    original_image: torch.Tensor,
    model: torch.nn.Module,
    target_class: int,
    preprocess_int: typing.Callable,
    backprocess_int: typing.Callable,
    preprocess_float: typing.Callable,
    max_iter: int,
) -> typing.Optional[torch.Tensor]:
    if not torch.cuda.is_available():
        raise Exception("cuda is required")
    model.eval().cuda()
    targets = torch.tensor([target_class]).cuda()

    original_image_cp = copy.deepcopy(original_image)

    criterion = torch.nn.CrossEntropyLoss()
    preprocessed_int_image = preprocess_int(original_image_cp.transpose(0, -1).cuda())
    preprocessed_float_image = F.convert_image_dtype(
        preprocessed_int_image, torch.float
    )
    preprocessed_float_image.requires_grad = True

    cur_class = (
        model(preprocess_float(preprocessed_float_image)[None, :]).argmax().item()
    )
    if cur_class == target_class:
        logging.info("current class matches target class")
        return original_image_cp

    optimizer_adv = torch.optim.Adam(
        [preprocessed_float_image], lr=1e-4, weight_decay=1e-3
    )

    i = 0
    while i < max_iter:
        outputs = model(preprocess_float(preprocessed_float_image)[None, :])
        if outputs.argmax().item() == target_class:
            logging.info(f"converged to adversarial example in {i} steps")
            return backprocess_int(preprocessed_float_image.to("cpu").detach())
        optimizer_adv.zero_grad()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer_adv.step()
        i += 1

    logging.info(f"failed to converge to adversarial example in {max_iter} iterations")
    return None
