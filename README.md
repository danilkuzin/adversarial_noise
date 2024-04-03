# Adversarial noise
This repository contains the adversarial noise generation example. The method is implemented in adversarial_noise.py file, example of running the method is in imagenet_example.py file.

## Method
The function receives an image and a target class. We backpropagate the gradients of the cross entropy loss with the target class w.r.t the input image, update it until the prediction matches the target class. A model for classification, its weights and inference transformations are assumed to be known. 

## Installation
### Conda
used to run MNIST example
```bash
conda create -n adversarial pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c conda-forge jupyterlab
conda install -c conda-forge matplotlib
```

### Docker
used for ImageNET example
```bash
sudo docker build --no-cache -t adversarial .
sudo docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it --rm -v $PWD:/source -v <path_to_dataset>:/datasets -v <path_to_results>:/results
```
