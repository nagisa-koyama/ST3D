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
# ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics
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

# nuscenes-devkit
RUN pip install nuscenes-devkit==1.0.5

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
# RUN apt install -y slurm-client iproute2 netcat

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
# RUN ln -s /storage/nuscenes_mini_v1_0 data/nuscenes
RUN ln -s /storage/nuscenes_full_v_1_0 data/nuscenes

# Add alias for test command
# RUN echo 'alias train_kitti="python train.py --cfg_file cfgs/kitti_models/second.yaml"' >> /root/.bashrc
# RUN echo 'alias train_lyft_multihead="python train.py --cfg_file cfgs/lyft_models/cbgs_second_multihead.yaml"' >> /root/.bashrc
# RUN echo 'alias train_pandaset="python train.py --cfg_file cfgs/pandaset_models/second.yaml"' >> /root/.bashrc
# RUN echo 'alias test_pandaset="python test.py --cfg_file cfgs/pandaset_models/secondiou_old_anchor.yaml --ckpt /storage/wandb/run-20230501_071704-iko5g53r/files/checkpoint_epoch_5.pth --batch_size 1"' >> /root/.bashrc
# RUN echo 'alias train_dev="bash scripts/dist_train.sh 2 --cfg_file cfgs/da-waymo-lyft-pandaset-to-kitti_models/domain_attention_head_per_dataset/second_old_anchor_kitti_waymo_lyft_pandaset_to_kitti.yaml --batch_size 40 --sync_bn"' >> /root/.bashrc
# RUN echo 'alias train_dev_wo_kitti="bash scripts/dist_train.sh 2 --cfg_file cfgs/da-waymo-lyft-pandaset-to-kitti_models/domain_attention_head_per_dataset/second_old_anchor_waymo_lyft_pandaset_to_kitti.yaml --batch_size 32 --sync_bn"' >> /root/.bashrc
# RUN echo 'alias train_lyft_naive="bash scripts/dist_train.sh 2 --cfg_file cfgs/da-waymo-lyft-pandaset-to-kitti_models/naive/second_old_anchor_lyft_to_lyft.yaml --batch_size 40 --sync_bn"' >> /root/.bashrc
# RUN echo 'alias train_pandaset_naive="bash scripts/dist_train.sh 2 --cfg_file cfgs/da-waymo-lyft-pandaset-to-kitti_models/naive/second_old_anchor_pandaset_to_pandaset.yaml --batch_size 40 --sync_bn"' >> /root/.bashrc
# RUN echo 'alias train_waymo_naive="bash scripts/dist_train.sh 2 --cfg_file cfgs/da-waymo-lyft-pandaset-to-kitti_models/naive/second_old_anchor_waymo_to_waymo.yaml --batch_size 40 --sync_bn"' >> /root/.bashrc
# RUN echo 'alias train_kitti_naive="bash scripts/dist_train.sh 2 --cfg_file cfgs/da-waymo-lyft-pandaset-to-kitti_models/naive/second_old_anchor_kitti_to_kitti.yaml --batch_size 40 --sync_bn"' >> /root/.bashrc
# RUN echo 'alias test_dev="bash scripts/dist_test.sh 2 --ckpt /storage/wandb/run-20230628_152333-84k9sy71/files/checkpoint_epoch_0.pth --cfg_file cfgs/da-waymo-lyft-pandaset-to-kitti_models/domain_attention_head_per_dataset/second_old_anchor_kitti_waymo_lyft_pandaset_to_kitti.yaml"' >> /root/.bashrc
# RUN echo 'alias test_kitti_head_per_dataset="python test.py --ckpt /storage/wandb/run-20230709_151045-x2rugf8v/files/checkpoint_epoch_10.pth --cfg_file cfgs/da-waymo-lyft-pandaset-to-kitti_models/domain_attention_head_per_dataset/second_old_anchor_kitti_waymo_lyft_pandaset_to_kitti.yaml"' >> /root/.bashrc
RUN echo 'alias test_all_dev="bash scripts/dist_test.sh 2 --ckpt_dir /storage/wandb/run-20230718_164733-mez7767c/files/ --eval_all --cfg_file cfgs/da-waymo-lyft-pandaset-to-kitti_models/domain_attention_head_per_dataset/second_old_anchor_kitti_waymo_lyft_pandaset_to_kitti.yaml"' >> /root/.bashrc
RUN echo 'alias test_all_dev_single="python test.py --ckpt_dir /storage/wandb/run-20230820_233910-s4xp5xnj/files/ --eval_all --cfg_file cfgs/da-waymo-lyft-pandaset-to-kitti_models/domain_attention_head_per_dataset/second_old_anchor_kitti_waymo_lyft_pandaset_to_kitti.yaml"' >> /root/.bashrc
RUN echo 'alias test_all_dev_single3="python test.py --ckpt_dir /storage/wandb/run-20230822_213309-pzetq062/files/ --eval_all --cfg_file cfgs/da-waymo-lyft-pandaset-to-kitti_models/domain_attention_head_per_dataset/second_old_anchor_kitti_waymo_lyft_pandaset_to_kitti.yaml"' >> /root/.bashrc
RUN echo 'alias test_all_dev_single_waymo_lyft_pandaset="python test.py --ckpt_dir /storage/wandb/run-20230910_132316-btsnvenk/files/ --eval_all --cfg_file cfgs/da-waymo-lyft-pandaset-to-kitti_models/domain_attention_head_per_dataset/second_old_anchor_waymo_lyft_pandaset_to_kitti.yaml"' >> /root/.bashrc
RUN echo 'alias test_all_dev_single4="python test.py --ckpt_dir /storage/wandb/run-20230401_133951-epu9c8tw/files/ --eval_all --cfg_file cfgs/da-waymo-lyft-pandaset-to-kitti_models/domain_attention/secondiou_old_anchor_kitti_waymo_lyft_pandaset_to_kitti.yaml"' >> /root/.bashrc
RUN echo 'alias test_all_dev_single5="python test.py --ckpt_dir /storage/wandb/run-20230827_163312-7w44bi3z/files/ --eval_all --cfg_file cfgs/da-waymo-lyft-pandaset-to-kitti_models/domain_attention/secondiou_old_anchor_kitti_waymo_lyft_pandaset_to_kitti.yaml"' >> /root/.bashrc
RUN echo 'alias train_dev_ckpt="bash scripts/dist_train.sh 2 --ckpt /storage/wandb/run-20230709_151045-x2rugf8v/files/checkpoint_epoch_10.pth --cfg_file cfgs/da-waymo-lyft-pandaset-to-kitti_models/domain_attention_head_per_dataset/second_old_anchor_kitti_waymo_lyft_pandaset_to_kitti.yaml --batch_size 40"' >> /root/.bashrc
RUN echo 'alias train_dev_ckpt_single="python train.py --ckpt /storage/wandb/run-20230709_151045-x2rugf8v/files/checkpoint_epoch_10.pth --cfg_file cfgs/da-waymo-lyft-pandaset-to-kitti_models/domain_attention_head_per_dataset/second_old_anchor_kitti_waymo_lyft_pandaset_to_kitti.yaml --batch_size 20"' >> /root/.bashrc
RUN echo 'alias test_lyft="python test.py --ckpt /storage/wandb/run-20230723_221713-2c94wfva/files/checkpoint_epoch_20.pth --cfg_file cfgs/da-waymo-lyft-pandaset-to-kitti_models/naive/secondiou_old_anchor_lyft_to_lyft.yaml"' >> /root/.bashrc
RUN echo 'alias self_train_single="python train.py --cfg_file cfgs/da-waymo-lyft-pandaset-to-kitti_models/domain_attention_head_per_dataset/second_old_anchor_waymo_lyft_pandaset_to_kitti_st3d_car_basebev.yaml --pretrained_model_teacher /storage/wandb/run-20230821_222115-469by94p/files/checkpoint_epoch_5.pth --pretrained_model /storage/wandb/run-20230410_165237-5qji6vhn/files/checkpoint_epoch_5.pth"' >> /root/.bashrc
RUN echo 'alias train_dev_wo_kitti_single="python train.py --cfg_file cfgs/da-waymo-lyft-pandaset-to-kitti_models/domain_attention_head_per_dataset/second_old_anchor_waymo_lyft_pandaset_to_kitti.yaml --batch_size 16 --sync_bn"' >> /root/.bashrc
RUN echo 'alias demo_kitti="python3 demo.py --cfg_file cfgs/da-waymo-lyft-pandaset-to-kitti_models/domain_attention_head_per_dataset/second_old_anchor_waymo_lyft_pandaset_to_kitti.yaml --ckpt /storage/wandb/run-20230821_222115-469by94p/files/checkpoint_epoch_5.pth"' >> /root/.bashrc
RUN echo 'alias demo_lyft="python3 demo.py --ckpt /storage/wandb/run-20230723_221713-2c94wfva/files/checkpoint_epoch_20.pth --cfg_file cfgs/da-waymo-lyft-pandaset-to-kitti_models/naive/second_old_anchor_lyft_to_lyft.yaml"' >> /root/.bashrc
RUN echo 'alias test_kitti_head_per_dataset="python test.py --cfg_file cfgs/da-waymo-lyft-pandaset-to-kitti_models/domain_attention_head_per_dataset/second_old_anchor_waymo_lyft_pandaset_to_kitti.yaml --ckpt /storage/wandb/run-20230821_222115-469by94p/files/checkpoint_epoch_5.pth"' >> /root/.bashrc
RUN echo 'alias train_nuscene="python train.py --cfg_file cfgs/nuscenes_models/second_car.yaml --epochs 3"' >> /root/.bashrc
RUN echo 'alias train_wlpn_head_per_dataset="python train.py --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/domain_attention_head_per_dataset/second_old_anchor_waymo_lyft_pandaset_nuscenes_car_ped.yaml --batch_size 16 --epochs 5"' >> /root/.bashrc
RUN echo 'alias train_lyft_head_per_dataset="python train.py --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/domain_attention_head_per_dataset/second_old_anchor_lyft_car_ped.yaml --batch_size 16 --epochs 20"' >> /root/.bashrc
RUN echo 'alias train_wlpn_single_head="python train.py --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/domain_attention/second_old_anchor_waymo_lyft_pandaset_nuscenes_car_ped.yaml --batch_size 16 --epochs 5"' >> /root/.bashrc
RUN echo 'alias train_wlpn_base_bev_single_head_no_pretrained="python train.py --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/domain_attention/second_old_anchor_base_bev_waymo_lyft_pandaset_nuscenes_car_ped.yaml --batch_size 16 --epochs 5"' >> /root/.bashrc
RUN echo 'alias train_wlpk_single_head="python train.py --cfg_file cfgs/da-waymo-lyft-pandaset-kitti-to-nuscenes_models/domain_attention/second_old_anchor_waymo_lyft_pandaset_kitti_car_ped.yaml --batch_size 16 --epochs 5"' >> /root/.bashrc
RUN echo 'alias train_wlpk_head_per_dataset="python train.py --cfg_file cfgs/da-waymo-lyft-pandaset-kitti-to-nuscenes_models/domain_attention_head_per_dataset/second_old_anchor_waymo_lyft_pandaset_kitti_car_ped.yaml --batch_size 16 --epochs 5"' >> /root/.bashrc
RUN echo 'alias test_wlpn_single_head="python test.py --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/domain_attention/second_old_anchor_waymo_lyft_pandaset_nuscenes_car_ped.yaml --ckpt_dir  /storage/wandb/run-20240205_161719-sw3p8xim/files --eval_all"' >> /root/.bashrc
RUN echo 'alias test_wlpn_single_head_sources="python test.py --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/domain_attention/second_old_anchor_waymo_lyft_pandaset_nuscenes_car_ped_sources.yaml --ckpt_dir  /storage/wandb/run-20240205_161719-sw3p8xim/files --eval_all"' >> /root/.bashrc
RUN echo 'alias test_wlpn_head_per_dataset="python test.py --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/domain_attention_head_per_dataset/second_old_anchor_waymo_lyft_pandaset_nuscenes_car_ped.yaml --ckpt_dir /storage/wandb/run-20240204_083626-wilfb1d0/files --eval_all"' >> /root/.bashrc
RUN echo 'alias test_wlpn_head_per_dataset_target="python test.py --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/domain_attention_head_per_dataset/second_old_anchor_waymo_lyft_pandaset_nuscenes_car_ped_target_kitti.yaml --ckpt_dir /storage/wandb/run-20240204_083626-wilfb1d0/files --eval_all"' >> /root/.bashrc
RUN echo 'alias test_wlpn_head_per_dataset_no_bt="python test.py --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/domain_attention_head_per_dataset/second_old_anchor_waymo_lyft_pandaset_nuscenes_car_ped.yaml --ckpt_dir /storage/wandb/run-20240204_084220-pbzdnjmg/files --eval_all"' >> /root/.bashrc
RUN echo 'alias self_train_wlpn_head_per_dataset_no_pretrained="python train.py --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/domain_attention_head_per_dataset/second_old_anchor_st3d_waymo_lyft_pandaset_nuscenes_to_kitti_car_ped.yaml --pretrained_model_teacher /storage/wandb/run-20240204_083626-wilfb1d0/files/checkpoint_epoch_5.pth --batch_size 16 --epochs 100"' >> /root/.bashrc
RUN echo 'alias self_train_lyft2kitti_head_per_dataset_no_pretrained="python train.py --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/domain_attention_head_per_dataset/second_old_anchor_st3d_lyft_to_kitti_car_ped.yaml --pretrained_model_teacher /storage/wandb/run-20240206_164927-ebrufbq5/files/checkpoint_epoch_5.pth --batch_size 16 --epochs 100"' >> /root/.bashrc
RUN echo 'alias self_train_wlpn_single_head_no_pretrained="python train.py --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/domain_attention/second_old_anchor_st3d_waymo_lyft_pandaset_nuscenes_to_kitti_car_ped.yaml --pretrained_model_teacher /storage/wandb/run-20240205_161719-sw3p8xim/files/checkpoint_epoch_5.pth --batch_size 16 --epochs 100"' >> /root/.bashrc
RUN echo 'alias conf_calib_wlpn_head_per_dataset="python test.py --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/domain_attention_head_per_dataset/second_old_anchor_conf_calib_waymo_lyft_pandaset_nuscenes_car_ped.yaml --ckpt /storage/wandb/run-20240204_083626-wilfb1d0/files/checkpoint_epoch_5.pth"' >> /root/.bashrc
RUN echo 'alias test_wlpn_head_per_dataset_conf_calibrated="python test.py --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/domain_attention_head_per_dataset/second_old_anchor_waymo_lyft_pandaset_nuscenes_conf_calibrated_car_ped.yaml --ckpt /storage/wandb/run-20240204_083626-wilfb1d0/files/checkpoint_epoch_5.pth"' >> /root/.bashrc
RUN echo 'alias self_train_wlpn_head_per_dataset_conf_calibrated_no_pretrained="python train.py --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/domain_attention_head_per_dataset/second_old_anchor_st3d_waymo_lyft_pandaset_nuscenes_to_kitti_conf_calibrated_car_ped.yaml --pretrained_model_teacher /storage/wandb/run-20240204_083626-wilfb1d0/files/checkpoint_epoch_5.pth --batch_size 16 --epochs 100"' >> /root/.bashrc
RUN echo 'alias test_wlpk_head_per_dataset_target="python test.py --cfg_file cfgs/da-waymo-lyft-pandaset-kitti-to-nuscenes_models/domain_attention_head_per_dataset/second_old_anchor_waymo_lyft_pandaset_kitti_car_ped_target_nuscenes.yaml --ckpt_dir /storage/wandb/run-20240211_000530-hng69v7k/files/ --eval_all"' >> /root/.bashrc
RUN echo 'alias self_train_wlpk_head_per_dataset_no_pretrained="python train.py --cfg_file cfgs/da-waymo-lyft-pandaset-kitti-to-nuscenes_models/domain_attention_head_per_dataset/second_old_anchor_st3d_waymo_lyft_pandaset_kitti_to_nuscenes_car_ped.yaml --pretrained_model_teacher /storage/wandb/run-20240211_000530-hng69v7k/files/checkpoint_epoch_5.pth --batch_size 16 --epochs 100"' >> /root/.bashrc
RUN echo 'alias self_train_wlpk_single_head_no_pretrained="python train.py --cfg_file cfgs/da-waymo-lyft-pandaset-kitti-to-nuscenes_models/domain_attention/second_old_anchor_st3d_waymo_lyft_pandaset_kitti_to_nuscenes_car_ped.yaml --pretrained_model_teacher /storage/wandb/run-20240222_142213-u2f6os5j/files/checkpoint_epoch_5.pth --batch_size 16 --epochs 100"' >> /root/.bashrc
RUN echo 'alias self_train_lyft2nuscenes_head_per_dataset_no_pretrained="python train.py --cfg_file cfgs/da-waymo-lyft-pandaset-kitti-to-nuscenes_models/domain_attention_head_per_dataset/second_old_anchor_st3d_lyft_to_nuscenes_car_ped.yaml --pretrained_model_teacher /storage/wandb/run-20240206_164927-ebrufbq5/files/checkpoint_epoch_5.pth --batch_size 16 --epochs 100"' >> /root/.bashrc
RUN echo 'alias test_lyft_head_per_dataset_target_kitti="python test.py --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/domain_attention_head_per_dataset/second_old_anchor_lyft_car_ped_target_kitti.yaml --ckpt_dir /storage/wandb/run-20240206_164927-ebrufbq5/files/ --eval_all"' >> /root/.bashrc
RUN echo 'alias test_lyft_head_per_dataset_target_nuscenes="python test.py --cfg_file cfgs/da-waymo-lyft-pandaset-kitti-to-nuscenes_models/domain_attention_head_per_dataset/second_old_anchor_lyft_car_ped_target_nuscenes.yaml --ckpt_dir /storage/wandb/run-20240206_164927-ebrufbq5/files/ --eval_all"' >> /root/.bashrc
RUN echo 'alias test_wlpn_head_per_dataset_conf_calibrated_target_kitti="python test.py --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/domain_attention_head_per_dataset/second_old_anchor_waymo_lyft_pandaset_nuscenes_conf_calibrated_car_ped_target_kitti.yaml --ckpt_dir /storage/wandb/run-20240204_083626-wilfb1d0/files --eval_all"' >> /root/.bashrc
RUN echo 'alias test_wlpk_single_head_target_nuscenes="python test.py --cfg_file cfgs/da-waymo-lyft-pandaset-kitti-to-nuscenes_models/domain_attention/second_old_anchor_waymo_lyft_pandaset_kitti_car_ped.yaml --ckpt_dir /storage/wandb/run-20240222_142213-u2f6os5j/files/ --eval_all"' >> /root/.bashrc
RUN echo 'alias self_train_wlpn_head_per_dataset_conf_calibrated_score_weighting_no_pretrained="python train.py --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/domain_attention_head_per_dataset/second_old_anchor_st3d_waymo_lyft_pandaset_nuscenes_to_kitti_conf_calibrated_score_weighting_car_ped.yaml --pretrained_model_teacher /storage/wandb/run-20240204_083626-wilfb1d0/files/checkpoint_epoch_5.pth --batch_size 16 --epochs 100"' >> /root/.bashrc
RUN echo 'alias conf_calib_wlpk_head_per_dataset="python test.py --cfg_file cfgs/da-waymo-lyft-pandaset-kitti-to-nuscenes_models/domain_attention_head_per_dataset/second_old_anchor_conf_calib_waymo_lyft_pandaset_kitti_car_ped.yaml --ckpt /storage/wandb/run-20240211_000530-hng69v7k/files/checkpoint_epoch_5.pth"' >> /root/.bashrc
RUN echo 'alias test_wlpk_head_per_dataset_conf_calibrated="python test.py --cfg_file cfgs/da-waymo-lyft-pandaset-kitti-to-nuscenes_models/domain_attention_head_per_dataset/second_old_anchor_waymo_lyft_pandaset_kitti_conf_calibrated_car_ped.yaml --ckpt_dir /storage/wandb/run-20240211_000530-hng69v7k/files --eval_all"' >> /root/.bashrc
RUN echo 'alias self_train_wlpk_head_per_dataset_conf_calibrated_no_pretrained="python train.py --cfg_file cfgs/da-waymo-lyft-pandaset-kitti-to-nuscenes_models/domain_attention_head_per_dataset/second_old_anchor_st3d_waymo_lyft_pandaset_kitti_to_nuscenes_conf_calibrated_car_ped.yaml --pretrained_model_teacher /storage/wandb/run-20240211_000530-hng69v7k/files/checkpoint_epoch_5.pth --batch_size 16 --epochs 100"' >> /root/.bashrc
RUN echo 'alias train_wlpk_base_bev_single_head_no_pretrained="python train.py --cfg_file cfgs/da-waymo-lyft-pandaset-kitti-to-nuscenes_models/domain_attention/second_old_anchor_base_bev_waymo_lyft_pandaset_kitti_car_ped.yaml --batch_size 16 --epochs 5"' >> /root/.bashrc
RUN echo 'alias self_train_wlpn_single_head_base_bev_no_pretrained="python train.py --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/domain_attention/second_old_anchor_st3d_base_bev_waymo_lyft_pandaset_nuscenes_to_kitti_car_ped.yaml --pretrained_model_teacher /storage/wandb/run-20240302_001912-ql4smy31/files/checkpoint_epoch_5.pth --batch_size 16 --epochs 100"' >> /root/.bashrc
RUN echo 'alias self_train_wlpk_single_head_base_bev_no_pretrained="python train.py --cfg_file cfgs/da-waymo-lyft-pandaset-kitti-to-nuscenes_models/domain_attention/second_old_anchor_st3d_base_bev_waymo_lyft_pandaset_kitti_to_nuscenes_car_ped.yaml --pretrained_model_teacher /storage/wandb/run-20240302_002102-6fdeg078/files/checkpoint_epoch_5.pth --batch_size 16 --epochs 100"' >> /root/.bashrc
RUN echo 'alias train_wlpn_base_bev_head_per_dataset="python train.py --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/domain_attention_head_per_dataset/second_old_anchor_base_bev_waymo_lyft_pandaset_nuscenes_car_ped.yaml --batch_size 16 --epochs 5"' >> /root/.bashrc
RUN echo 'alias train_wlpk_base_bev_head_per_dataset="python train.py --cfg_file cfgs/da-waymo-lyft-pandaset-kitti-to-nuscenes_models/domain_attention_head_per_dataset/second_old_anchor_base_bev_waymo_lyft_pandaset_kitti_car_ped.yaml --batch_size 16 --epochs 5"' >> /root/.bashrc
RUN echo 'alias self_train_wlpn_head_per_dataset_no_pretrained="python train.py --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/domain_attention_head_per_dataset/second_old_anchor_st3d_waymo_lyft_pandaset_nuscenes_to_kitti_car_ped.yaml --pretrained_model_teacher /storage/wandb/run-20240204_083626-wilfb1d0/files/checkpoint_epoch_5.pth --batch_size 16 --epochs 100"' >> /root/.bashrc
RUN echo 'alias self_train_wlpn_head_per_dataset_pretrained="python train.py --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/domain_attention_head_per_dataset/second_old_anchor_st3d_waymo_lyft_pandaset_nuscenes_to_kitti_car_ped.yaml --pretrained_model_teacher /storage/wandb/run-20240204_083626-wilfb1d0/files/checkpoint_epoch_5.pth --pretrained_model /storage/wandb/run-20240204_083626-wilfb1d0/files/checkpoint_epoch_5.pth --batch_size 16 --epochs 100"' >> /root/.bashrc
RUN echo 'alias self_train_wlpk_head_per_dataset_pretrained="python train.py --cfg_file cfgs/da-waymo-lyft-pandaset-kitti-to-nuscenes_models/domain_attention_head_per_dataset/second_old_anchor_st3d_waymo_lyft_pandaset_kitti_to_nuscenes_car_ped.yaml --pretrained_model_teacher /storage/wandb/run-20240211_000530-hng69v7k/files/checkpoint_epoch_5.pth --pretrained_model /storage/wandb/run-20240211_000530-hng69v7k/files/checkpoint_epoch_5.pth --batch_size 16 --epochs 100"' >> /root/.bashrc
RUN echo 'alias self_train_wlpn_basebev_head_per_dataset_no_pretrained="python train.py --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/domain_attention_head_per_dataset/second_old_anchor_st3d_basebev_waymo_lyft_pandaset_nuscenes_to_kitti_car_ped.yaml --pretrained_model_teacher /storage/wandb/run-20240311_160001-ki04bs16/files/checkpoint_epoch_5.pth --batch_size 16 --epochs 100"' >> /root/.bashrc
RUN echo 'alias train_lyft_basebev_head_per_dataset="python train.py --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/domain_attention_head_per_dataset/second_old_anchor_basebev_lyft_car_ped.yaml --batch_size 16 --epochs 20"' >> /root/.bashrc
RUN echo 'alias conf_calib_wlpn_basebev_head_per_dataset="python test.py --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/domain_attention_head_per_dataset/second_old_anchor_conf_calib_basebev_waymo_lyft_pandaset_nuscenes_car_ped.yaml --ckpt /storage/wandb/run-20240311_160001-ki04bs16/files/checkpoint_epoch_5.pth"' >> /root/.bashrc
RUN echo 'alias self_train_lyft2kitti_basebev_head_per_dataset_no_pretrained="python train.py --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/domain_attention_head_per_dataset/second_old_anchor_st3d_basebev_lyft_to_kitti_car_ped.yaml --pretrained_model_teacher /storage/wandb/run-20240313_162824-gcskf29r/files/checkpoint_epoch_5.pth --batch_size 16 --epochs 100"' >> /root/.bashrc
RUN echo 'alias self_train_wlpk_basebev_head_per_dataset_no_pretrained="python train.py --cfg_file cfgs/da-waymo-lyft-pandaset-kitti-to-nuscenes_models/domain_attention_head_per_dataset/second_old_anchor_st3d_basebev_waymo_lyft_pandaset_kitti_to_nuscenes_car_ped.yaml --pretrained_model_teacher /storage/wandb/run-20240311_160015-x22b4i0s/files/checkpoint_epoch_5.pth --batch_size 16 --epochs 100"' >> /root/.bashrc
RUN echo 'alias self_train_lyft2nuscenes_basebev_head_per_dataset_no_pretrained="python train.py --cfg_file cfgs/da-waymo-lyft-pandaset-kitti-to-nuscenes_models/domain_attention_head_per_dataset/second_old_anchor_st3d_basebev_lyft_to_nuscenes_car_ped.yaml --pretrained_model_teacher /storage/wandb/run-20240313_162824-gcskf29r/files/checkpoint_epoch_5.pth --batch_size 16 --epochs 100"' >> /root/.bashrc
RUN echo 'alias test_wlpn_basebev_head_per_dataset_target="python test.py --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/domain_attention_head_per_dataset/second_old_anchor_basebev_waymo_lyft_pandaset_nuscenes_car_ped_target_kitti.yaml --ckpt_dir /storage/wandb/run-20240311_160001-ki04bs16/files --eval_all"' >> /root/.bashrc
RUN echo 'alias test_wlpk_basebev_head_per_dataset_target="python test.py --cfg_file cfgs/da-waymo-lyft-pandaset-kitti-to-nuscenes_models/domain_attention_head_per_dataset/second_old_anchor_basebev_waymo_lyft_pandaset_kitti_car_ped_target_nuscenes.yaml --ckpt_dir /storage/wandb/run-20240311_160015-x22b4i0s/files/ --eval_all"' >> /root/.bashrc
RUN echo 'alias self_train_wlpn_basebev_head_per_dataset_conf_calibrated_no_pretrained="python train.py --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/domain_attention_head_per_dataset/second_old_anchor_st3d_basebev_waymo_lyft_pandaset_nuscenes_to_kitti_conf_calibrated_car_ped.yaml --pretrained_model_teacher /storage/wandb/run-20240311_160001-ki04bs16/files/checkpoint_epoch_5.pth --batch_size 16 --epochs 100"' >> /root/.bashrc
RUN echo 'alias conf_calib_wlpk_basebev_head_per_dataset="python test.py --cfg_file cfgs/da-waymo-lyft-pandaset-kitti-to-nuscenes_models/domain_attention_head_per_dataset/second_old_anchor_conf_calib_basebev_waymo_lyft_pandaset_kitti_car_ped.yaml --ckpt /storage/wandb/run-20240311_160015-x22b4i0s/files/checkpoint_epoch_5.pth"' >> /root/.bashrc
RUN echo 'alias test_lyft_basebev_head_per_dataset_target_kitti="python test.py --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/domain_attention_head_per_dataset/second_old_anchor_basebev_lyft_car_ped_target_kitti.yaml --ckpt /storage/wandb/run-20240313_162824-gcskf29r/files/checkpoint_epoch_5.pth"' >> /root/.bashrc
RUN echo 'alias test_lyft_basebev_head_per_dataset_target_nuscenes="python test.py --cfg_file cfgs/da-waymo-lyft-pandaset-kitti-to-nuscenes_models/domain_attention_head_per_dataset/second_old_anchor_basebev_lyft_car_ped_target_nuscenes.yaml --ckpt /storage/wandb/run-20240313_162824-gcskf29r/files/checkpoint_epoch_5.pth"' >> /root/.bashrc

#RUN nohup Xvfb -ac ${DISPLAY} -screen 0 1280x780x24 &
# ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

# Copy latest ST3D source from local storage except folders in .dockerignore.
# Assuming that directory structure is kept.
COPY . $WORK_DIR/ST3D
WORKDIR $WORK_DIR/ST3D
RUN git log -n 1

WORKDIR $WORK_DIR/ST3D/tools
