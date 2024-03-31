FROM nvcr.io/nvidia/pytorch:24.02-py3

COPY ./requirements.txt ./requirements.txt

RUN python -m pip install --upgrade pip && pip install -r requirements.txt

RUN python -c 'from huggingface_hub._login import _login; _login(token="", add_to_git_credential=False)'

WORKDIR /source
