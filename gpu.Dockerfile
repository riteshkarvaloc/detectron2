FROM nvidia/cuda:10.1-cudnn7-devel

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
        python3-opencv ca-certificates python3-dev python3-pip git wget sudo ninja-build unzip nano

#RUN ln -sv /usr/bin/python3 /usr/bin/python

ENV PATH="/home/dkube/.local/bin:${PATH}"
RUN wget https://bootstrap.pypa.io/get-pip.py && \
        python3 get-pip.py --user && \
        rm get-pip.py

# Installing jupyterlab
RUN pip3 install jupyterlab && \
    pip3 install jupyterlab[extras]

# Installing jupyterlab
RUN pip3 install jupyterlab && \
    pip3 install jupyterlab[extras]

# install dependencies
# See https://pytorch.org/ for other options if you use a different version of CUDA
RUN pip3 install --user tensorboard cmake   # cmake from apt-get is too old
RUN pip3 install --user torch==1.7.1 torchvision==0.8.2 -f https://download.pytorch.org/whl/cu101/torch_stable.html

RUN pip3 install --user 'git+https://github.com/facebookresearch/fvcore'
# install detectron2
RUN git clone https://github.com/facebookresearch/detectron2 detectron2_repo
# set FORCE_CUDA because during `docker build` cuda is not accessible
ENV FORCE_CUDA="1"
# This will by default build detectron2 for all common cuda architectures and take a lot more time,
# because inside `docker build`, there is no way to tell which architecture will be used.
ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

RUN pip3 install --user -e detectron2_repo

# Set a fixed model cache directory.
ENV FVCORE_CACHE="/tmp"
#WORKDIR /home/dkube/detectron2_repo
RUN useradd -m dkube && echo "dkube:dkube" | chpasswd && adduser dkube sudo
RUN usermod -aG sudo,root dkube
RUN echo 'dkube ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

LABEL heritage="dkube"
RUN python3 -m ipykernel install --user
COPY scripts/jupyter/bashrc /etc/bash.bashrc
RUN jupyter notebook --generate-config
RUN git config --global user.email "dkube@oneconvergence.com"
RUN git config --global user.name "dkube"
ENV DKUBE_NB_ARGS ""
CMD ["sh", "-c", "jupyter lab --ip=0.0.0.0 --port=8888 --allow-root $DKUBE_NB_ARGS"]