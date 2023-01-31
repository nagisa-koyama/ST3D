#####################################################################
# This Dockerfile was generated by dockerfile-generator.sh
# Image build:
#   docker build --force-rm=true --rm=true -t {REPOSITORY}:{TAG} --no-cache=true .
# Container build:
#   docker run --runtim=nvidia --rm -it -u {USER ID}:{GROUP ID} -p {PORT NUM}:8888 -v {local dir}:/home/{USER NAME} --ipc=host --name {CONTAINER NAME} {REPOSITORY}:{TAG}
#####################################################################

# =========================================
# Set Versions
# =========================================
FROM nvidia/cuda:11.6.0-cudnn8-devel-ubuntu18.04
#FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
#FROM scrin/dev-spconv
ARG PYTHON_VERSION=python3.7
ARG TORCH_VERSION=1.9.0

ENV LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH}
ENV CPATH=/usr/local/include:${CPATH}
ENV CUDA_PATH=/usr/local/cuda
ENV CPATH=${CUDA_PATH}/include:${CPATH}
ENV LD_LIBRARY_PATH=${CUDA_PATH}/lib64:${CUDA_PATH}/lib:${LD_LIBRARY_PATH}
ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_DRIVER_CAPABILITIES=all
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
                                               vim\
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

RUN apt-get install -y libboost-all-dev
RUN apt remove -y cmake
RUN wget https://cmake.org/files/v3.13/cmake-3.13.2-Linux-x86_64.sh
RUN sh cmake-3.13.2-Linux-x86_64.sh --skip-license --include-subdir
RUN export PATH=$PATH:/root/cmake-3.13.2-Linux-x86_64/bin
RUN ln -s /root/cmake-3.13.2-Linux-x86_64/bin/* /usr/bin/
RUN cmake --version

RUN apt install -y \
build-essential libbz2-dev libdb-dev \
libffi-dev libgdbm-dev liblzma-dev \
libncursesw5-dev libreadline-dev libsqlite3-dev \
libssl-dev tk-dev uuid-dev \
zlib1g-dev \
emacs

#RUN wget https://www.python.org/ftp/python/3.6.9/Python-3.6.9.tgz
#RUN wget https://www.python.org/ftp/python/3.7.3/Python-3.7.3.tgz
#RUN tar -xvf Python-3.6.9.tgz
#RUN tar -xvf Python-3.7.3.tgz
#WORKDIR $WORK_DIR/Python-3.6.9
#WORKDIR $WORK_DIR/Python-3.7.3
#RUN ./configure
#RUN make
#RUN make install

#RUN apt install -y python3.7

WORKDIR $WORK_DIR
RUN apt install -y python-pip python3-pip
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install wheel
# RUN wget https://download.pytorch.org/whl/cu90/torch-1.1.0-cp36-cp36m-linux_x86_64.whl
#RUN wget https://download.pytorch.org/whl/cu100/torch-1.1.0-cp37-cp37m-linux_x86_64.whl
# RUN pip3 install torch-1.1.0-cp36-cp36m-linux_x86_64.whl
#RUN python3 -m pip install torch-1.1.0-cp37-cp37m-linux_x86_64.whl
#RUN pip install torch===1.6.0 -f https://download.pytorch.org/whl/torch_stable.html
#RUN pip install torch===${TORCH_VERSION}+ -f https://download.pytorch.org/whl/torch_stable.html
# RUN pip install torch===1.12.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html
RUN python3 -m pip install torch==1.12.0+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

WORKDIR $WORK_DIR
#RUN git clone https://github.com/traveller59/spconv.git --recursive
RUN git clone https://github.com/nagisa-koyama/spconv.git --recursive
WORKDIR $WORK_DIR/spconv/
#RUN git checkout v1.2.1
RUN git checkout v1.2.1_commentout
#RUN git checkout 8da6f967fb9a054d8870c3515b1b44eca2103634
RUN cmake --version
RUN cd ./third_party/pybind11/ && git submodule update --init
RUN python3 setup.py bdist_wheel
RUN cd ./dist && pip3 install spconv*.whl

ARG ST3D_BRANCH=v20230131_pcdet0.6
WORKDIR $WORK_DIR
#RUN git clone https://github.com/CVMI-Lab/ST3D.git --recursive
RUN git clone https://github.com/nagisa-koyama/ST3D.git --recursive
WORKDIR $WORK_DIR/ST3D
RUN git fetch --all -p #redo
RUN git checkout ${ST3D_BRANCH}
#RUN cd ./pcdet && git submodule update --init
#RUN git checkout st3d++
#RUN git checkout st3d_v0.2

#RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r requirements.txt
RUN git branch
RUN git log -n 1
RUN python3 setup.py develop

# Configurations for wandb and waymo open.
RUN python3 -m pip install waymo-open-dataset-tf-2-5-0 wandb
ENV WANDB_API_KEY 8f252267771b0b737b6b5bcfce56c9e52dc50a99
ENV WANDB_PROJECT st3d
ENV CUDA_VISIBLE_DEVICES 0,1

RUN python3 -m pip install -U numpy==1.18.5
RUN python3 -m pip install pyyaml==5.4.1

# Configurations for saving image with mayavi.
RUN python3 -m pip install mayavi pyqt5
RUN apt update && apt install -y libxkbcommon-x11-0 libxkb-* libxcb-* xvfb
ENV DISPLAY :1
RUN nohup Xvfb -ac ${DISPLAY} -screen 0 1280x780x24 &

RUN git config --global user.email "nagisa.koyama@gmail.com"
RUN git config --global user.name "Nagisa Koyama"
RUN git config --global core.editor 'emacs -nw'

ADD "https://www.random.org/cgi-bin/randbyte?nbytes=10&format=h" /dev/null
RUN git fetch --all\
 && git reset --hard origin/${ST3D_BRANCH}\
 && git log -n 1
RUN mv data/waymo data/waymo_orig
RUN mv data/kitti data/kitti_orig
RUN ln -s /storage/waymo_open_dataset_v_1_4_0/pcdet_structure/ data/waymo
RUN ln -s /storage/kitti/ data/kitti

RUN python3 -m pip install -U lyft_dataset_sdk==0.0.8
RUN mv data/lyft data/lyft_orig
RUN ln -s /mnt/disk3/koyama/level5-3d-object-detection data/lyft
