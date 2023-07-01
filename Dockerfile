# =========================================
# Set Versions
# =========================================
FROM nvidia/cuda:11.6.0-cudnn8-devel-ubuntu18.04
ARG PYTHON_VERSION=python3.7

ENV CPATH=/usr/local/include:${CPATH}
ENV CUDA_PATH=/usr/local/cuda
ENV CPATH=${CUDA_PATH}/include:${CPATH}
ENV LD_LIBRARY_PATH=${CUDA_PATH}/lib64:${CUDA_PATH}/lib:/usr/local/lib:${LD_LIBRARY_PATH}
ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV CUDA_VISIBLE_DEVICES 0,1
ENV WANDB_API_KEY 8f252267771b0b737b6b5bcfce56c9e52dc50a99
ENV WANDB_PROJECT st3d
ENV WORK_DIR=/root/${USER_NAME}
WORKDIR $WORK_DIR

# =========================================
# Ubuntu setting
# =========================================
RUN rm -rf /var/lib/apt/lists/*\
            /etc/apt/source.list.d/cuda.list\
            /etc/apt/source.list.d/nvidia-ml.list

RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
RUN apt-get update -y &&\
    apt-get upgrade -y
 
RUN apt-get install -y --no-install-recommends build-essential\
                                               apt-utils\
                                               ca-certificates\
                                               make\
                                               wget\
                                               git\
                                               curl\
                                               emacs\
                                               openssh-server

# =========================================
# Python setting
# =========================================
RUN apt-get update\
 && apt-get install unzip\
 && apt-get install -y software-properties-common\
 && add-apt-repository ppa:deadsnakes/ppa\
 && apt-get update\
 && apt-get install -y ${PYTHON_VERSION} ${PYTHON_VERSION}-dev python3-distutils-extra\
 && wget -O ~/get-pip.py https://bootstrap.pypa.io/get-pip.py\
 && ${PYTHON_VERSION} ~/get-pip.py\
 && ln -s /usr/bin/${PYTHON_VERSION} /usr/local/bin/python3\
 && ln -s /usr/bin/${PYTHON_VERSION} /usr/local/bin/python

# pytorch
WORKDIR $WORK_DIR
RUN apt install -y python-pip python3-pip
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install wheel
RUN python3 -m pip install torch==1.12.0+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

# wandb
RUN python3 -m pip install wandb

# Configurations for saving image with mayavi.
RUN python3 -m pip install mayavi pyqt5
RUN apt update && apt install -y libxkbcommon-x11-0 libxkb-* libxcb-* xvfb

# cmake, which is needed by spconv1.2
WORKDIR $WORK_DIR
RUN apt-get install -y libboost-all-dev
RUN apt remove -y cmake
RUN wget https://cmake.org/files/v3.13/cmake-3.13.2-Linux-x86_64.sh
RUN sh cmake-3.13.2-Linux-x86_64.sh --skip-license --include-subdir
RUN export PATH=$PATH:/root/cmake-3.13.2-Linux-x86_64/bin
RUN ln -s /root/cmake-3.13.2-Linux-x86_64/bin/* /usr/bin/
RUN cmake --version

# spconv1.2
RUN git clone https://github.com/nagisa-koyama/spconv.git --recursive
WORKDIR $WORK_DIR/spconv/
RUN git checkout v1.2.1_commentout
RUN cmake --version
RUN cd ./third_party/pybind11/ && git submodule update --init
RUN python3 setup.py bdist_wheel
RUN cd ./dist && pip3 install spconv*.whl

# spconv2.0
# RUN python3 -m pip install spconv-cu116

# Waymo and lyft open dataset.
RUN python3 -m pip install -U waymo-open-dataset-tf-2-5-0
RUN python3 -m pip install -U lyft_dataset_sdk==0.0.8

# pandaset-devkit
WORKDIR $WORK_DIR
# RUN git clone https://github.com/scaleapi/pandaset-devkit.git
RUN git clone https://github.com/lea-v/pandaset-devkit.git
WORKDIR $WORK_DIR/pandaset-devkit
RUN git checkout feature/addPointCloudTransformations
WORKDIR $WORK_DIR/pandaset-devkit/python
RUN python3 -m pip install .

# ST3D dependency install
ARG ST3D_INSTALLABLE_BRANCH=v20230204_refactor
WORKDIR $WORK_DIR
#RUN git clone https://github.com/CVMI-Lab/ST3D.git --recursive
RUN git clone https://github.com/nagisa-koyama/ST3D.git --recursive
WORKDIR $WORK_DIR/ST3D
RUN git fetch --all -p
RUN git checkout ${ST3D_INSTALLABLE_BRANCH}
#RUN cd ./pcdet && git submodule update --init
RUN python3 -m pip install -r requirements.txt
RUN python3 setup.py develop

# Additional pips
RUN python3 -m pip install torchinfo

# Additional apt
RUN apt install -y slurm-client iproute2 netcat

# ST3D branch update
# WORKDIR $WORK_DIR/ST3D
# ADD "https://www.random.org/cgi-bin/randbyte?nbytes=10&format=h" /dev/null
# ARG ST3D_DEV_BRANCH=v20230319_port_DA_module
# RUN git fetch --all\
#   && git reset --hard origin/${ST3D_DEV_BRANCH}\
#   && git log -n 1

# Storage linking
WORKDIR $WORK_DIR/ST3D
RUN mv data/waymo data/waymo_orig
RUN mv data/kitti data/kitti_orig
RUN mv data/lyft data/lyft_orig
RUN ln -s /storage/waymo_open_dataset_v_1_4_0/pcdet_structure/ data/waymo
RUN ln -s /storage/kitti/ data/kitti
RUN ln -s /storage/level5-3d-object-detection data/lyft
RUN ln -s /storage/pandaset data/pandaset

# Add alias for test command
RUN echo 'alias train_kitti="python train.py --cfg_file cfgs/kitti_models/second.yaml"' >> /root/.bashrc
RUN echo 'alias train_lyft_multihead="python train.py --cfg_file cfgs/lyft_models/cbgs_second_multihead.yaml"' >> /root/.bashrc
RUN echo 'alias train_pandaset="python train.py --cfg_file cfgs/pandaset_models/second.yaml"' >> /root/.bashrc
RUN echo 'alias test_pandaset="python test.py --cfg_file cfgs/pandaset_models/secondiou_old_anchor.yaml --ckpt /storage/wandb/run-20230501_071704-iko5g53r/files/checkpoint_epoch_5.pth --batch_size 1"' >> /root/.bashrc
RUN echo 'alias train_dev="bash scripts/dist_train.sh 2 --cfg_file cfgs/da-waymo-lyft-pandaset-to-kitti_models/domain_attention_head_per_dataset/second_old_anchor_kitti_waymo_lyft_pandaset_to_kitti.yaml"' >> /root/.bashrc
RUN echo 'alias test_dev="python test.py --ckpt /storage/wandb/run-20230617_165237-ynybp1v7/files/checkpoint_epoch_5.pth --cfg_file cfgs/da-waymo-lyft-pandaset-to-kitti_models/domain_attention_head_per_dataset/second_old_anchor_kitti_waymo_lyft_pandaset_to_kitti.yaml"' >> /root/.bashrc
# Copy latest ST3D source from local storage except folders in .dockerignore.
# Assuming that directory structure is kept.
COPY . $WORK_DIR/ST3D
WORKDIR $WORK_DIR/ST3D
RUN git log -n 1

WORKDIR $WORK_DIR/ST3D/tools
