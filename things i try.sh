echo "מצבי עבודה
1. identity - \"test on train \" sanity
2. add augmentation
3. retiming + augmentation
output videos:
1. global/local
2. me / someone else"


conda activate caroline

# train non augmented
python "$PWD/everybodydancenow/train_fullts.py" \
    --name noam_model_global_non_augmented \
    --dataroot "$PWD/output/noam_slight_left_ang_right.mp4/everybodydancenow/train_dataset/" \
    --checkpoints_dir "$PWD/output/noam_slight_left_ang_right.mp4/ckpts/everybodydancenow/" \
    --loadSize 512 \
    --no_instance \
    --resize_or_crop none \
    --no_flip \
    --tf_log \
    --label_nc 6 \
    --niter 4 \
    --niter_decay 1

python "$PWD/everybodydancenow/train_fullts.py" \
--name noam_model_local_non_augmented \
--dataroot "$PWD/output/noam_slight_left_ang_right.mp4/everybodydancenow/train_dataset/" \
--checkpoints_dir "$PWD/output/noam_slight_left_ang_right.mp4/ckpts/everybodydancenow/" \
--load_pretrain "$PWD/output/noam_slight_left_ang_right.mp4/ckpts/everybodydancenow/noam_model_global_non_augmented" \
--netG local \
--ngf 32 \
--num_D 3 \
--resize_or_crop none \
--no_instance \
--no_flip \
--tf_log \
--label_nc 6 \
--niter 4 \
--niter_decay 1


# train augmented
python "$PWD/everybodydancenow/train_fullts.py" \
    --name noam_model_global_augmented \
    --dataroot "$PWD/output/noam_slight_left_ang_right.mp4/everybodydancenow/train_dataset/" \
    --checkpoints_dir "$PWD/output/noam_slight_left_ang_right.mp4/ckpts/everybodydancenow/" \
    --loadSize 512 \
    --no_instance \
    --resize_or_crop resize_and_crop \
    --no_flip \
    --tf_log \
    --label_nc 6 \
    --niter 4 \
    --niter_decay 1


CUDA_VISIBLE_DEVICES=1 python "$PWD/everybodydancenow/train_fullts.py" \
--name noam_model_local_augmented \
--dataroot "$PWD/output/noam_slight_left_ang_right.mp4/everybodydancenow/train_dataset/" \
--checkpoints_dir "$PWD/output/noam_slight_left_ang_right.mp4/ckpts/everybodydancenow/" \
--load_pretrain "$PWD/output/noam_slight_left_ang_right.mp4/ckpts/everybodydancenow/noam_model_global_augmented" \
--netG local \
--ngf 32 \
--num_D 3 \
--resize_or_crop resize_and_crop \
--no_instance \
--no_flip \
--tf_log \
--label_nc 6 \
--niter 4 \
--niter_decay 1


CUDA_VISIBLE_DEVICES=1 python "$PWD/everybodydancenow/train_fullts.py" \
--name noam_model_local_augmented_15 \
--dataroot "$PWD/output/noam_slight_left_ang_right.mp4/everybodydancenow/train_dataset/" \
--checkpoints_dir "$PWD/output/noam_slight_left_ang_right.mp4/ckpts/everybodydancenow/" \
--load_pretrain "$PWD/output/noam_slight_left_ang_right.mp4/ckpts/everybodydancenow/noam_model_global_augmented" \
--netG local \
--ngf 32 \
--num_D 3 \
--resize_or_crop resize_and_crop \
--no_instance \
--no_flip \
--tf_log \
--label_nc 6 \
--niter 8 \
--niter_decay 2


# create validation (not used)
CUDA_VISIBLE_DEVICES=1 python "$PWD/everybodydancenow/data_prep/graph_avesmooth.py" \
--keypoints_dir "$PWD/output/noam_slight_left_ang_right.mp4/openpose_json_output/" \
--frames_dir "$PWD/output/noam_slight_left_ang_right.mp4/frames/" \
--save_dir "$PWD/output/noam_slight_left_ang_right.mp4/everybodydancenow/val_dataset/" \
--spread 500 29999 4 \
--facetexts




# test
CUDA_VISIBLE_DEVICES=0 python "$PWD/everybodydancenow/test_fullts.py" \
--name noam_model_global_augmented \
--dataroot "$PWD/output/noam_slight_left_ang_right.mp4/everybodydancenow/train_dataset/" \
--checkpoints_dir "$PWD/output/noam_slight_left_ang_right.mp4/ckpts/everybodydancenow/" \
--results_dir "$PWD/output/noam_slight_left_ang_right.mp4/results/test_on_train/" \
--loadSize 512 \
--no_instance \
--how_many 10000 \
--label_nc 6 \
--resize_or_crop resize_and_crop \
--no_instance \
--no_flip 


CUDA_VISIBLE_DEVICES=1 python "$PWD/everybodydancenow/test_fullts.py" \
--name noam_model_global_non_augmented \
--dataroot "$PWD/output/noam_slight_left_ang_right.mp4/everybodydancenow/train_dataset/" \
--checkpoints_dir "$PWD/output/noam_slight_left_ang_right.mp4/ckpts/everybodydancenow/" \
--results_dir "$PWD/output/noam_slight_left_ang_right.mp4/results/test_on_train/" \
--loadSize 512 \
--no_instance \
--resize_or_crop none \
--how_many 10000 \
--label_nc 6

CUDA_VISIBLE_DEVICES=1 python "$PWD/everybodydancenow/test_fullts.py" \
--name noam_model_local_augmented \
--dataroot "$PWD/output/noam_slight_left_ang_right.mp4/everybodydancenow/train_dataset/" \
--checkpoints_dir "$PWD/output/noam_slight_left_ang_right.mp4/ckpts/everybodydancenow/" \
--results_dir "$PWD/output/noam_slight_left_ang_right.mp4/results/test_on_train/" \
--netG local \
--ngf 32 \
--loadSize 512 \
--how_many 10000 \
--label_nc 6 \
--resize_or_crop resize_and_crop \
--no_instance \
--no_flip 

CUDA_VISIBLE_DEVICES=1 python "$PWD/everybodydancenow/test_fullts.py" \
--name noam_model_local_non_augmented \
--dataroot "$PWD/output/noam_slight_left_ang_right.mp4/everybodydancenow/train_dataset/" \
--checkpoints_dir "$PWD/output/noam_slight_left_ang_right.mp4/ckpts/everybodydancenow/" \
--results_dir "$PWD/output/noam_slight_left_ang_right.mp4/results/test_on_train/" \
--netG local \
--ngf 32 \
--loadSize 512 \
--no_instance \
--resize_or_crop none \
--how_many 10000 \
--label_nc 6




# make videos
ffmpeg -framerate 30 -start_number 0 -y \
	-i "output/noam_slight_left_ang_right.mp4/results/test_on_train/noam_model_local_augmented/test_latest/images/%03d_synthesized_image.png" \
    "output/noam_slight_left_ang_right.mp4/results/test_on_train/noam_model_local_augmented.mp4"

ffmpeg -framerate 30 -start_number 0 -y \
	-i "output/noam_slight_left_ang_right.mp4/results/test_on_train/noam_model_global_augmented/test_latest/images/%03d_synthesized_image.png" \
	   "output/noam_slight_left_ang_right.mp4/results/test_on_train/noam_model_global_augmented.mp4"

ffmpeg -framerate 30 -start_number 0 -y \
	-i "output/noam_slight_left_ang_right.mp4/results/test_on_train/noam_model_local_non_augmented/test_latest/images/%03d_synthesized_image.png" \
	   "output/noam_slight_left_ang_right.mp4/results/test_on_train/noam_model_local_non_augmented.mp4"

ffmpeg -framerate 30 -start_number 0 -y \
	-i "output/noam_slight_left_ang_right.mp4/results/test_on_train/noam_model_global_non_augmented/test_latest/images/%03d_synthesized_image.png" \
	   "output/noam_slight_left_ang_right.mp4/results/test_on_train/noam_model_global_non_augmented.mp4"




ffmpeg -framerate 30 -start_number 0 -y \
	-i "output/noam_slight_left_ang_right.mp4/results/test_on_train/noam_model_local_non_augmented_15/test_latest/images/%03d_synthesized_image.png" \
	   "output/noam_slight_left_ang_right.mp4/results/test_on_train/noam_model_local_non_augmented_15.mp4"
       
ffmpeg -framerate 30 -start_number 0 -y \
	-i "output/noam_slight_left_ang_right.mp4/results/test_on_train/noam_model_local_augmented_15/test_latest/images/%03d_synthesized_image.png" \
	   "output/noam_slight_left_ang_right.mp4/results/test_on_train/noam_model_local_augmented_15.mp4"






# test front
CUDA_VISIBLE_DEVICES=0 python "$PWD/everybodydancenow/test_fullts.py" \
--name noam_model_global_augmented \
--dataroot "$PWD/output/noam_front.mp4/everybodydancenow/test_dataset/" \
--checkpoints_dir "$PWD/output/noam_slight_left_ang_right.mp4/ckpts/everybodydancenow/" \
--results_dir "$PWD/output/noam_slight_left_ang_right.mp4/results/test_on_front/" \
--loadSize 512 \
--no_instance \
--how_many 10000 \
--label_nc 6


CUDA_VISIBLE_DEVICES=1 python "$PWD/everybodydancenow/test_fullts.py" \
--name noam_model_global_non_augmented \
--dataroot "$PWD/output/noam_front.mp4/everybodydancenow/test_dataset/" \
--checkpoints_dir "$PWD/output/noam_slight_left_ang_right.mp4/ckpts/everybodydancenow/" \
--results_dir "$PWD/output/noam_slight_left_ang_right.mp4/results/test_on_front/" \
--loadSize 512 \
--no_instance \
--how_many 10000 \
--label_nc 6


# make videos

ffmpeg -framerate 30 -start_number 0 -y \
	-i "output/noam_slight_left_ang_right.mp4/results/test_on_front/noam_model_global_augmented/test_latest/images/%03d_synthesized_image.png" \
	   "output/noam_slight_left_ang_right.mp4/results/test_on_front/noam_front_model_global_augmented.mp4"


ffmpeg -framerate 30 -start_number 0 -y \
	-i "output/noam_slight_left_ang_right.mp4/results/test_on_front/noam_model_global_non_augmented/test_latest/images/%03d_synthesized_image.png" \
    "output/noam_slight_left_ang_right.mp4/results/test_on_front/noam_front_model_local_augmented.mp4"


