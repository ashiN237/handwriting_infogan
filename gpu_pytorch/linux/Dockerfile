FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel

# avoid gpg error
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub

# apt-get
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update -qq && \
    apt-get install -y curl git sudo tmux vim wget libglib2.0-0 libsm6 libxrender1 libxext6 libgl1-mesa-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /var/cache/apk/* 

# python
ENV WORK_PATH /workspace
WORKDIR $WORK_PATH
ENV PYTHONPATH "/workspace:${PYTHONPATH}"
ENV PIP_OPTIONS "--no-cache-dir --progress-bar off"
COPY ./gpu_pytorch/linux/requirements.txt $WORK_PATH/gpu_pytorch/linux/
RUN pip install ${PIP_OPTIONS} -r $WORK_PATH/gpu_pytorch/linux/requirements.txt && \
        pip install ${PIP_OPTIONS} -U setuptools

# duser setting
ARG USER_ID
ARG GROUP_ID
RUN addgroup --gid $GROUP_ID duser && \
    adduser --disabled-password --gecos '' --uid ${USER_ID} --gid ${GROUP_ID} duser && \
    adduser duser sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER duser