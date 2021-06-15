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

python "$PWD/everybodydancenow/train_fullts.py" \
--name noam_model_local_non_augmented_15 \
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
--niter 8 \
--niter_decay 2


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


CUDA_VISIBLE_DEVICES=1 python "$PWD/everybodydancenow/train_fullts.py" \
    --name noam_model_global_augmented_small \
    --dataroot "$PWD/output/noam_slight_left_ang_right.mp4/everybodydancenow/train_dataset/" \
    --checkpoints_dir "$PWD/output/noam_slight_left_ang_right.mp4/ckpts/everybodydancenow/" \
    --loadSize 256 \
    --fineSize 128 \
    --no_instance \
    --resize_or_crop resize_and_crop \
    --no_flip \
    --tf_log \
    --label_nc 6 \
    --niter 4 \
    --niter_decay 1

CUDA_VISIBLE_DEVICES=1 python "$PWD/everybodydancenow/train_fullts.py" \
--name noam_model_local_augmented_small \
--dataroot "$PWD/output/noam_slight_left_ang_right.mp4/everybodydancenow/train_dataset/" \
--checkpoints_dir "$PWD/output/noam_slight_left_ang_right.mp4/ckpts/everybodydancenow/" \
--load_pretrain "$PWD/output/noam_slight_left_ang_right.mp4/ckpts/everybodydancenow/noam_model_global_augmented_small" \
--loadSize 512 \
--fineSize 256 \
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
--resize_or_crop resize \
--how_many 10000 \
--label_nc 6

CUDA_VISIBLE_DEVICES=0 python "$PWD/everybodydancenow/test_fullts.py" \
--name noam_model_local_augmented \
--dataroot "$PWD/output/noam_slight_left_ang_right.mp4/everybodydancenow/train_dataset/" \
--checkpoints_dir "$PWD/output/noam_slight_left_ang_right.mp4/ckpts/everybodydancenow/" \
--results_dir "$PWD/output/noam_slight_left_ang_right.mp4/results/test_on_train/" \
--netG local \
--ngf 32 \
--loadSize 128 \
--how_many 10000 \
--label_nc 6 \
--resize_or_crop resize \
--no_instance \
--no_flip 

CUDA_VISIBLE_DEVICES=0 python "$PWD/everybodydancenow/test_fullts.py" \
--name noam_model_local_non_augmented \
--dataroot "$PWD/output/noam_slight_left_ang_right.mp4/everybodydancenow/train_dataset/" \
--checkpoints_dir "$PWD/output/noam_slight_left_ang_right.mp4/ckpts/everybodydancenow/" \
--results_dir "$PWD/output/noam_slight_left_ang_right.mp4/results/test_on_train/" \
--netG local \
--ngf 32 \
--loadSize 512 \
--no_instance \
--resize_or_crop resize \
--how_many 10000 \
--label_nc 6

CUDA_VISIBLE_DEVICES=1 python "$PWD/everybodydancenow/test_fullts.py" \
--name noam_model_local_augmented_small \
--dataroot "$PWD/output/noam_slight_left_ang_right.mp4/everybodydancenow/train_dataset/" \
--checkpoints_dir "$PWD/output/noam_slight_left_ang_right.mp4/ckpts/everybodydancenow/" \
--results_dir "$PWD/output/noam_slight_left_ang_right.mp4/results/test_on_train/" \
--netG local \
--ngf 32 \
--loadSize 512 \
--how_many 10000 \
--label_nc 6 \
--resize_or_crop resize \
--no_instance \
--no_flip 



# make videos
ffmpeg -framerate 30 -start_number 0 -y \
	-i "output/noam_slight_left_ang_right.mp4/results/test_on_train/noam_model_local_augmented/test_latest/images/%03d_synthesized_image.png" \
        -c:v copy \
    "output/noam_slight_left_ang_right.mp4/results/test_on_train/noam_model_local_augmented.mkv"

ffmpeg -framerate 30 -start_number 0 -y \
	-i "output/noam_slight_left_ang_right.mp4/results/test_on_train/noam_model_global_augmented/test_latest/images/%03d_synthesized_image.png" \
        -c:v copy \
	   "output/noam_slight_left_ang_right.mp4/results/test_on_train/noam_model_global_augmented.mkv"

ffmpeg -framerate 30 -start_number 0 -y \
	-i "output/noam_slight_left_ang_right.mp4/results/test_on_train/noam_model_local_non_augmented/test_latest/images/%03d_synthesized_image.png" \
        -c:v copy \
	   "output/noam_slight_left_ang_right.mp4/results/test_on_train/noam_model_local_non_augmented.mkv"

ffmpeg -framerate 30 -start_number 0 -y \
	-i "output/noam_slight_left_ang_right.mp4/results/test_on_train/noam_model_global_non_augmented/test_latest/images/%03d_synthesized_image.png" \
        -c:v copy \
	   "output/noam_slight_left_ang_right.mp4/results/test_on_train/noam_model_global_non_augmented.mkv"




ffmpeg -framerate 30 -start_number 0 -y \
	-i "output/noam_slight_left_ang_right.mp4/results/test_on_train/noam_model_local_non_augmented_15/test_latest/images/%03d_synthesized_image.png" \
        -c:v copy \
	   "output/noam_slight_left_ang_right.mp4/results/test_on_train/noam_model_local_non_augmented_15.mkv"
       
ffmpeg -framerate 30 -start_number 0 -y \
	-i "output/noam_slight_left_ang_right.mp4/results/test_on_train/noam_model_local_augmented_15/test_latest/images/%03d_synthesized_image.png" \
        -c:v copy \
	   "output/noam_slight_left_ang_right.mp4/results/test_on_train/noam_model_local_augmented_15.mkv"

ffmpeg -framerate 30 -start_number 0 -y \
	-i "output/noam_slight_left_ang_right.mp4/results/test_on_train/noam_model_local_augmented_small/test_latest/images/%03d_synthesized_image.png" \
        -c:v copy \
	   "output/noam_slight_left_ang_right.mp4/results/test_on_train/noam_model_local_augmented_small.mkv"






# test front
CUDA_VISIBLE_DEVICES=0 python "$PWD/everybodydancenow/test_fullts.py" \
--name noam_model_global_non_augmented \
--dataroot "$PWD/output/noam_front.mp4/everybodydancenow/test_dataset/" \
--checkpoints_dir "$PWD/output/noam_slight_left_ang_right.mp4/ckpts/everybodydancenow/" \
--results_dir "$PWD/output/noam_slight_left_ang_right.mp4/results/test_on_front/" \
--loadSize 1024 \
--no_instance \
--how_many 10000 \
--label_nc 6


CUDA_VISIBLE_DEVICES=1 python "$PWD/everybodydancenow/test_fullts.py" \
--name noam_model_local_non_augmented \
--dataroot "$PWD/output/noam_front.mp4/everybodydancenow/test_dataset/" \
--checkpoints_dir "$PWD/output/noam_slight_left_ang_right.mp4/ckpts/everybodydancenow/" \
--results_dir "$PWD/output/noam_slight_left_ang_right.mp4/results/test_on_front/" \
--netG local \
--ngf 32 \
--loadSize 512 \
--no_instance \
--resize_or_crop resize \
--how_many 10000 \
--label_nc 6



CUDA_VISIBLE_DEVICES=1 python "$PWD/everybodydancenow/test_fullts.py" \
--name noam_model_local_augmented_small \
--dataroot "$PWD/output/noam_front.mp4/everybodydancenow/train_dataset/" \
--checkpoints_dir "$PWD/output/noam_slight_left_ang_right.mp4/ckpts/everybodydancenow/" \
--results_dir "$PWD/output/noam_slight_left_ang_right.mp4/results/test_on_front/" \
--netG local \
--ngf 32 \
--loadSize 512 \
--how_many 10000 \
--label_nc 6 \
--resize_or_crop resize \
--no_instance \
--no_flip 

# make videos

ffmpeg -framerate 30 -start_number 0 -y \
	-i "output/noam_slight_left_ang_right.mp4/results/test_on_front/noam_model_local_non_augmented/test_latest/images/%03d_synthesized_image.png" \
        -c:v copy \
	   "output/noam_slight_left_ang_right.mp4/results/test_on_front/noam_model_local_non_augmented.mkv"

ffmpeg -framerate 30 -start_number 0 -y \
	-i "output/noam_slight_left_ang_right.mp4/results/test_on_front/noam_model_global_non_augmented/test_latest/images/%03d_synthesized_image.png" \
        -c:v copy \
	   "output/noam_slight_left_ang_right.mp4/results/test_on_front/noam_model_global_non_augmented.mkv"

ffmpeg -framerate 30 -start_number 0 -y \
	-i "output/noam_slight_left_ang_right.mp4/results/test_on_front/noam_model_local_augmented_small/test_latest/images/%03d_synthesized_image.png" \
        -c:v copy \
    "output/noam_slight_left_ang_right.mp4/results/test_on_front/noam_model_local_augmented_small.mkv"


# test dor

CUDA_VISIBLE_DEVICES=o python "$PWD/everybodydancenow/test_fullts.py" \
--name noam_model_local_augmented_small \
--dataroot "$PWD/output/dor_front.mp4/everybodydancenow/train_dataset/" \
--checkpoints_dir "$PWD/output/noam_slight_left_ang_right.mp4/ckpts/everybodydancenow/" \
--results_dir "$PWD/output/noam_slight_left_ang_right.mp4/results/test_on_dor_front/" \
--netG local \
--ngf 32 \
--loadSize 512 \
--how_many 10000 \
--label_nc 6 \
--resize_or_crop resize \
--no_instance \
--no_flip 

CUDA_VISIBLE_DEVICES=1 python "$PWD/everybodydancenow/test_fullts.py" \
--name noam_model_local_non_augmented \
--dataroot "$PWD/output/dor_front.mp4/everybodydancenow/train_dataset/" \
--checkpoints_dir "$PWD/output/noam_slight_left_ang_right.mp4/ckpts/everybodydancenow/" \
--results_dir "$PWD/output/noam_slight_left_ang_right.mp4/results/test_on_dor_front/" \
--netG local \
--ngf 32 \
--loadSize 512 \
--no_instance \
--resize_or_crop resize \
--how_many 10000 \
--label_nc 6


# make videos

ffmpeg -framerate 30 -start_number 0 -y \
	-i "output/noam_slight_left_ang_right.mp4/results/test_on_dor_front/noam_model_local_non_augmented/test_latest/images/%03d_synthesized_image.png" \
	   "output/noam_slight_left_ang_right.mp4/results/test_on_dor_front/noam_model_local_non_augmented.mp4"


ffmpeg -framerate 30 -start_number 0 -y \
	-i "output/noam_slight_left_ang_right.mp4/results/test_on_dor_front/noam_model_local_augmented_small/test_latest/images/%03d_synthesized_image.png" \
    "output/noam_slight_left_ang_right.mp4/results/test_on_dor_front/noam_model_local_augmented_small.mp4"



# test 20210203_174949.mp4
CUDA_VISIBLE_DEVICES=0 python "$PWD/everybodydancenow/test_fullts.py" \
--name noam_model_local_augmented_small \
--dataroot "$PWD/output/20210203_174949.mp4/everybodydancenow/train_dataset/" \
--checkpoints_dir "$PWD/output/noam_slight_left_ang_right.mp4/ckpts/everybodydancenow/" \
--results_dir "$PWD/output/noam_slight_left_ang_right.mp4/results/test_on_20210203_174949/" \
--netG local \
--ngf 32 \
--loadSize 512 \
--how_many 10000 \
--label_nc 6 \
--resize_or_crop resize \
--no_instance \
--no_flip 

CUDA_VISIBLE_DEVICES=1 python "$PWD/everybodydancenow/test_fullts.py" \
--name noam_model_local_non_augmented \
--dataroot "$PWD/output/20210203_174949.mp4/everybodydancenow/train_dataset/" \
--checkpoints_dir "$PWD/output/noam_slight_left_ang_right.mp4/ckpts/everybodydancenow/" \
--results_dir "$PWD/output/noam_slight_left_ang_right.mp4/results/test_on_20210203_174949/" \
--netG local \
--ngf 32 \
--loadSize 512 \
--no_instance \
--resize_or_crop resize \
--how_many 10000 \
--label_nc 6


# make videos
ffmpeg -framerate 30 -start_number 0 -y \
	-i "output/noam_slight_left_ang_right.mp4/results/test_on_20210203_174949/noam_model_local_non_augmented/test_latest/images/%03d_synthesized_image.png" \
        -c:v copy \
	   "output/noam_slight_left_ang_right.mp4/results/test_on_20210203_174949/noam_model_local_non_augmented.mkv"

ffmpeg -framerate 30 -start_number 0 -y \
	-i "output/noam_slight_left_ang_right.mp4/results/test_on_20210203_174949/noam_model_local_augmented_small/test_latest/images/%03d_synthesized_image.png" \
        -c:v copy \
    "output/noam_slight_left_ang_right.mp4/results/test_on_20210203_174949/noam_model_local_augmented_small.mkv"

ffmpeg -i videos/20210203_174949.mp4 -vf scale=1024:-1 output/noam_slight_left_ang_right.mp4/results/test_on_20210203_174949/scaled.mp4
ffmpeg \
    -i output/noam_slight_left_ang_right.mp4/results/test_on_20210203_174949/scaled.mp4 \
    -i output/noam_slight_left_ang_right.mp4/results/test_on_20210203_174949/noam_model_local_non_augmented.mkv \
    -i output/noam_slight_left_ang_right.mp4/results/test_on_20210203_174949/noam_model_local_augmented_small.mkv \
    -filter_complex vstack=inputs=3 "output/noam_slight_left_ang_right.mp4/results/test_on_20210203_174949/output.mkv"



# test 20210203_175137.mp4
CUDA_VISIBLE_DEVICES=0 python "$PWD/everybodydancenow/test_fullts.py" \
--name noam_model_local_augmented_small \
--dataroot "$PWD/output/20210203_175137.mp4/everybodydancenow/train_dataset/" \
--checkpoints_dir "$PWD/output/noam_slight_left_ang_right.mp4/ckpts/everybodydancenow/" \
--results_dir "$PWD/output/noam_slight_left_ang_right.mp4/results/test_on_20210203_175137/" \
--netG local \
--ngf 32 \
--loadSize 512 \
--how_many 10000 \
--label_nc 6 \
--resize_or_crop resize \
--no_instance \
--no_flip 

CUDA_VISIBLE_DEVICES=1 python "$PWD/everybodydancenow/test_fullts.py" \
--name noam_model_local_non_augmented \
--dataroot "$PWD/output/20210203_175137.mp4/everybodydancenow/train_dataset/" \
--checkpoints_dir "$PWD/output/noam_slight_left_ang_right.mp4/ckpts/everybodydancenow/" \
--results_dir "$PWD/output/noam_slight_left_ang_right.mp4/results/test_on_20210203_175137/" \
--netG local \
--ngf 32 \
--loadSize 512 \
--no_instance \
--resize_or_crop resize \
--how_many 10000 \
--label_nc 6


# make videos
ffmpeg -framerate 30 -start_number 0 -y \
	-i "output/noam_slight_left_ang_right.mp4/results/test_on_20210203_175137/noam_model_local_non_augmented/test_latest/images/%03d_synthesized_image.png" \
        -c:v copy \
	   "output/noam_slight_left_ang_right.mp4/results/test_on_20210203_175137/noam_model_local_non_augmented.mkv"

ffmpeg -framerate 30 -start_number 0 -y \
	-i "output/noam_slight_left_ang_right.mp4/results/test_on_20210203_175137/noam_model_local_augmented_small/test_latest/images/%03d_synthesized_image.png" \
        -c:v copy \
    "output/noam_slight_left_ang_right.mp4/results/test_on_20210203_175137/noam_model_local_augmented_small.mkv"

ffmpeg -i videos/20210203_175137.mp4 -vf scale=1024:-1 output/noam_slight_left_ang_right.mp4/results/test_on_20210203_175137/scaled.mp4
ffmpeg \
    -i output/noam_slight_left_ang_right.mp4/results/test_on_20210203_175137/scaled.mp4 \
    -i output/noam_slight_left_ang_right.mp4/results/test_on_20210203_175137/noam_model_local_non_augmented.mkv \
    -i output/noam_slight_left_ang_right.mp4/results/test_on_20210203_175137/noam_model_local_augmented_small.mkv \
    -filter_complex vstack=inputs=3 "output/noam_slight_left_ang_right.mp4/results/test_on_20210203_175137/output.mkv"


# test 20210203_175137.mp4 bigger
CUDA_VISIBLE_DEVICES=1 python "$PWD/everybodydancenow/test_fullts.py" \
--name noam_model_local_augmented_small \
--dataroot "$PWD/output/20210203_175137.mp4/everybodydancenow/train_dataset/" \
--checkpoints_dir "$PWD/output/noam_slight_left_ang_right.mp4/ckpts/everybodydancenow/" \
--results_dir "$PWD/output/noam_slight_left_ang_right.mp4/results/test_on_20210203_175137_bigger/" \
--netG local \
--ngf 32 \
--loadSize 416 \
--how_many 10000 \
--label_nc 6 \
--resize_or_crop resize \
--no_instance \
--no_flip 

CUDA_VISIBLE_DEVICES=1 python "$PWD/everybodydancenow/test_fullts.py" \
--name noam_model_local_non_augmented \
--dataroot "$PWD/output/20210203_175137.mp4/everybodydancenow/train_dataset/" \
--checkpoints_dir "$PWD/output/noam_slight_left_ang_right.mp4/ckpts/everybodydancenow/" \
--results_dir "$PWD/output/noam_slight_left_ang_right.mp4/results/test_on_20210203_175137_bigger/" \
--netG local \
--ngf 32 \
--loadSize 416 \
--no_instance \
--resize_or_crop resize \
--how_many 10000 \
--label_nc 6


# make videos
ffmpeg -framerate 30 -start_number 0 -y \
	-i "output/noam_slight_left_ang_right.mp4/results/test_on_20210203_175137_bigger/noam_model_local_non_augmented/test_latest/images/%03d_synthesized_image.png" \
        -c:v copy \
	   "output/noam_slight_left_ang_right.mp4/results/test_on_20210203_175137_bigger/noam_model_local_non_augmented.mkv"

ffmpeg -framerate 30 -start_number 0 -y \
	-i "output/noam_slight_left_ang_right.mp4/results/test_on_20210203_175137_bigger/noam_model_local_augmented_small/test_latest/images/%03d_synthesized_image.png" \
        -c:v copy \
    "output/noam_slight_left_ang_right.mp4/results/test_on_20210203_175137_bigger/noam_model_local_augmented_small.mkv"

ffmpeg -i videos/20210203_175137.mp4 -vf scale=832:-1 output/noam_slight_left_ang_right.mp4/results/test_on_20210203_175137_bigger/scaled.mp4
ffmpeg \
    -i output/noam_slight_left_ang_right.mp4/results/test_on_20210203_175137_bigger/scaled.mp4 \
    -i output/noam_slight_left_ang_right.mp4/results/test_on_20210203_175137_bigger/noam_model_local_non_augmented.mkv \
    -i output/noam_slight_left_ang_right.mp4/results/test_on_20210203_175137_bigger/noam_model_local_augmented_small.mkv \
    -filter_complex vstack=inputs=3 "output/noam_slight_left_ang_right.mp4/results/test_on_20210203_175137_bigger/output.mkv"







# test noam_side_to_side.mp4
CUDA_VISIBLE_DEVICES=0 python "$PWD/everybodydancenow/test_fullts.py" \
--name noam_model_local_augmented_small \
--dataroot "$PWD/output/noam_side_to_side.mp4/everybodydancenow/train_dataset/" \
--checkpoints_dir "$PWD/output/noam_slight_left_ang_right.mp4/ckpts/everybodydancenow/" \
--results_dir "$PWD/output/noam_slight_left_ang_right.mp4/results/test_on_noam_side_to_side/" \
--netG local \
--ngf 32 \
--loadSize 512 \
--how_many 10000 \
--label_nc 6 \
--resize_or_crop resize \
--no_instance \
--no_flip 

CUDA_VISIBLE_DEVICES=1 python "$PWD/everybodydancenow/test_fullts.py" \
--name noam_model_local_non_augmented \
--dataroot "$PWD/output/noam_side_to_side.mp4/everybodydancenow/train_dataset/" \
--checkpoints_dir "$PWD/output/noam_slight_left_ang_right.mp4/ckpts/everybodydancenow/" \
--results_dir "$PWD/output/noam_slight_left_ang_right.mp4/results/test_on_noam_side_to_side/" \
--netG local \
--ngf 32 \
--loadSize 512 \
--no_instance \
--resize_or_crop resize \
--how_many 10000 \
--label_nc 6


# make videos

# fill missing frames

num=1563
# $(ls output/noam_slight_left_ang_right.mp4/results/test_on_noam_side_to_side/noam_model_local_non_augmented/test_latest/images/ | tail -n 1 | grep -oP '[1-9]\d+')

for i in $(seq 1 $num)
do
    currfile="output/noam_slight_left_ang_right.mp4/results/test_on_noam_side_to_side/noam_model_local_non_augmented/test_latest/images/$(printf '%03d' $i)_synthesized_image.png"

    if [ ! -f "$currfile" ]; then
        for j in $(seq $i $num)
        do
            nextfile="output/noam_slight_left_ang_right.mp4/results/test_on_noam_side_to_side/noam_model_local_non_augmented/test_latest/images/$(printf '%03d' $j)_synthesized_image.png"
            
            if test -f "$nextfile"; then
                    echo "cp $j $i"
                    cp $nextfile $currfile
                    break
            fi
        done
    fi
done
for i in $(seq 1 $num)
do
    currfile="output/noam_slight_left_ang_right.mp4/results/test_on_noam_side_to_side/noam_model_local_augmented_small/test_latest/images/$(printf '%03d' $i)_synthesized_image.png"

    if [ ! -f "$currfile" ]; then
        for j in $(seq $i $num)
        do
            nextfile="output/noam_slight_left_ang_right.mp4/results/test_on_noam_side_to_side/noam_model_local_augmented_small/test_latest/images/$(printf '%03d' $j)_synthesized_image.png"
            if test -f "$nextfile"; then
                    echo "cp $j $i"
                    cp $nextfile $currfile
                    break
            fi
        done
    fi
done
# end fill missing frames



ffmpeg -framerate 30 -start_number 0 -y \
	-i "output/noam_slight_left_ang_right.mp4/results/test_on_noam_side_to_side/noam_model_local_non_augmented/test_latest/images/%03d_synthesized_image.png" \
        -c:v copy \
	   "output/noam_slight_left_ang_right.mp4/results/test_on_noam_side_to_side/noam_model_local_non_augmented.mkv"

ffmpeg -framerate 30 -start_number 0 -y \
	-i "output/noam_slight_left_ang_right.mp4/results/test_on_noam_side_to_side/noam_model_local_augmented_small/test_latest/images/%03d_synthesized_image.png" \
        -c:v copy \
    "output/noam_slight_left_ang_right.mp4/results/test_on_noam_side_to_side/noam_model_local_augmented_small.mkv"

ffmpeg -i videos/noam_side_to_side.mp4 -vf scale=1024:-1 output/noam_slight_left_ang_right.mp4/results/test_on_noam_side_to_side/scaled.mp4
ffmpeg \
    -i output/noam_slight_left_ang_right.mp4/results/test_on_noam_side_to_side/scaled.mp4 \
    -i output/noam_slight_left_ang_right.mp4/results/test_on_noam_side_to_side/noam_model_local_non_augmented.mkv \
    -i output/noam_slight_left_ang_right.mp4/results/test_on_noam_side_to_side/noam_model_local_augmented_small.mkv \
    -filter_complex vstack=inputs=3 "output/noam_slight_left_ang_right.mp4/results/test_on_noam_side_to_side/output.mkv"