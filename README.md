# Adversarial
## Installation
```bash
conda create -n adversarial pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c conda-forge jupyterlab
conda install -c conda-forge matplotlib
```

## Docker
sudo docker build --no-cache -t adversarial .
sudo docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it --rm -v $PWD:/source -v /media/DanilOlyaMark/Linux_Data_Disk-15/datasets:/datasets adversarial