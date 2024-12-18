FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04
# Python 3.10.12

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    curl \
    ssh \
    git \
    nano \
    python3 python3-pip \
    python-is-python3 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U pip \
    && pip install --no-cache-dir \
    numpy \
    torch \
    torchvision \
    pymupdf \
    pillow \
    matplotlib

ARG USERNAME=developer
ARG GROUPNAME=developers
ARG UID
ARG GID
ARG PASSWORD=developer
RUN groupadd -g $GID $GROUPNAME && \
    useradd -m -s /bin/bash -u $UID -g $GID -G sudo $USERNAME && \
    echo $USERNAME:$PASSWORD | chpasswd && \
    echo "$USERNAME   ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
USER $USERNAME

WORKDIR /workspace/handwrittentextgen
ENV FORCE_CUDA="1"
