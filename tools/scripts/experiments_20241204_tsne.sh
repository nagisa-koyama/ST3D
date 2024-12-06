# TSNE with the dann simple 64 target model
python tsne.py --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/head_per_dataset_with_source/second_old_anchor_st3d_basebev_lyft_to_kitti_source_target_eval_car_ped.yaml --ckpt /storage/wandb/run-20241204_134810-q3yip843/files/checkpoint_epoch_100.pth --out_filename tsne_q3yip843.png -platform offscreen

# TSNE with the dann simple 512 target model
#python tsne.py --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/head_per_dataset_with_source/second_old_anchor_st3d_basebev_lyft_to_kitti_source_target_eval_car_ped.yaml --ckpt /storage/wandb/run-20241201_161453-lgt39ghv/files/checkpoint_epoch_100.pth --out_filename tsne_lgt49ghv.png -platform offscreen

# TSNE with the dann target model
#python tsne.py --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/head_per_dataset_with_source/second_old_anchor_st3d_basebev_lyft_to_kitti_source_target_eval_car_ped.yaml --ckpt /storage/wandb/run-20241110_103232-g4qujukf/files/checkpoint_epoch_100.pth --out_filename tsne_g4qujukf.png -platform offscreen

# TSNE with the dann source model
#python tsne.py --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/head_per_dataset_with_source/second_old_anchor_st3d_basebev_lyft_to_kitti_source_target_eval_car_ped.yaml --ckpt /storage/wandb/run-20241111_160513-x3i6rtqe/files/checkpoint_epoch_100.pth --out_filename tsne_x3i6rtqe.png -platform offscreen

# TSNE with the target model
python tsne.py --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/head_per_dataset_with_source/second_old_anchor_st3d_basebev_lyft_to_kitti_source_target_eval_car_ped.yaml --ckpt /storage/wandb/run-20241031_161514-nfzq3i87/files/checkpoint_epoch_100.pth --out_filename tsne_nfzq3i87.png -platform offscreen

# TSNE with the source model
#python tsne.py --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/head_per_dataset_with_source/second_old_anchor_st3d_basebev_lyft_to_kitti_source_target_eval_car_ped.yaml --ckpt /storage/wandb/run-20241104_161410-e354coth/files/checkpoint_epoch_100.pth --out_file tsne_e354coth.png -platform offscreen

# TSNE with the dann model
#python tsne.py --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/head_per_dataset_with_source/second_old_anchor_st3d_basebev_lyft_to_kitti_source_target_eval_car_ped.yaml --ckpt /storage/wandb/run-20241110_145136-5zep34i6/files/checkpoint_epoch_100.pth --out_file tsne_5zep34i6.png -platform offscreen

# TSNE with the dann simple 512 source target model
python tsne.py --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/head_per_dataset_with_source/second_old_anchor_st3d_basebev_lyft_to_kitti_source_target_eval_car_ped.yaml --ckpt /storage/wandb/run-20241202_163316-38qwzblr/files/checkpoint_epoch_100.pth --out_filename tsne_38qwzblr.png -platform offscreen

