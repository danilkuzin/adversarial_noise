import copy
import logging
import typing

import torch
from torchvision.transforms import functional as F


def generate_adversarial_image(
    original_image: torch.Tensor,
    model: torch.nn.Module,
    target_class: int,
    preprocess: typing.Callable,
    max_iter: int,
) -> typing.Optional[torch.Tensor]:
    """
    This function generates an adversarial example for given model and image. 
    The adversarial example is found by changing input via backpropagation.
    It uses the transformations from the original inference process used for the model, 
    implemented via kornia library that allows differentiable transformations.

    original_image: input image for adversarial example, dimensions are [Width Height Cnannel]. Uint8
    model: torchvision classification neural network
    target_class: target output for adversarial example
    preprocess: function that performs transformations used in model inference. This implementation should be differentiable.
    max_iter: maximal number of iterations that the optimizer is allowed to make to find change the input image

    returns adversarial example, if it is found. None otherwise
    """
    if not torch.cuda.is_available():
        raise Exception("cuda is required")
    model.eval().cuda()
    targets = torch.tensor([target_class]).cuda()

    original_image_cp = copy.deepcopy(original_image)

    criterion = torch.nn.CrossEntropyLoss()
    float_image = F.convert_image_dtype(
        original_image_cp.transpose(0, -1).cuda(), torch.float
    )
    float_image.requires_grad = True

    cur_class = model(preprocess(float_image[None, :])).argmax().item()
    if cur_class == target_class:
        logging.info("current class matches target class")
        return original_image_cp

    optimizer_adv = torch.optim.Adam(
        [float_image], lr=1e-4, weight_decay=1e-3
    )

    i = 0
    while i < max_iter:
        outputs = model(preprocess(float_image[None, :]))
        optimizer_adv.zero_grad()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer_adv.step()

        if outputs.argmax().item() == target_class:
            logging.info(f"converged to adversarial example in {i} steps")
            return (
                float_image.to("cpu").detach()
            ).transpose(0, -1)

        i += 1

    logging.info(f"failed to converge to adversarial example in {max_iter} iterations")
    return None
