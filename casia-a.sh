FILENAME=casiaa_fyc
FILENAME=casiaa_xxj

docker run -v "$PWD/output/$FILENAME:/noam" --gpus=all -it --rm -e NVIDIA_VISIBLE_DEVICES=0 cwaffles/openpose ./build/examples/openpose/openpose.bin -image_dir ../noam/frames -hand -disable_blending -face -output_resolution 320x-1 -number_people_max 1 -no_gui_verbose --render_pose 0 -display 0 --write_json ../noam/openpose_json_output/

python "/media/noam/HDD 3TB/create_dataset/libs/everybodydancenow/data_prep/graph_train.py" \
--keypoints_dir "$PWD/output/$FILENAME/openpose_json_output/"                        --frames_dir "$PWD/output/$FILENAME/frames/"  \
--save_dir "$PWD/output/$FILENAME/everybodydancenow/train_dataset/"                  --spread 0 6331 1                --facetexts


cp -r "$PWD/output/$FILENAME/everybodydancenow/train_dataset/train_label" "$PWD/output/$FILENAME/everybodydancenow/train_dataset/test_label"

# inference on noam_slight_left_ang_right

CUDA_VISIBLE_DEVICES=1 python "$PWD/everybodydancenow/test_fullts.py" \
--name noam_model_local_non_augmented \
--dataroot "$PWD/output/$FILENAME/everybodydancenow/train_dataset/" \
--checkpoints_dir "$PWD/output/noam_slight_left_ang_right.mp4/ckpts/everybodydancenow/" \
--results_dir "$PWD/output/noam_slight_left_ang_right.mp4/results/casia/$FILENAME/" \
--netG local \
--ngf 32 \
--loadSize 512 \
--no_instance \
--resize_or_crop resize \
--how_many 10000 \
--label_nc 6


ffmpeg -framerate 30 -start_number 0 -y \
	-i "output/noam_slight_left_ang_right.mp4/results/casia/$FILENAME/noam_model_local_non_augmented/test_latest/images/%03d_synthesized_image.png" \
        -c:v copy \
	   "output/noam_slight_left_ang_right.mp4/results/casia/$FILENAME/noam_model_local_non_augmented.mkv"

ffmpeg -framerate 30 -start_number 0 -y \
	-i "output/00_1/frames/%03d.png" \
        -c:v copy \
	   "output/noam_slight_left_ang_right.mp4/results/casia/$FILENAME/original.mkv"


# train casia model

python "$PWD/everybodydancenow/train_fullts.py" \
    --name noam_model_global_non_augmented \
    --dataroot "$PWD/output/$FILENAME/everybodydancenow/train_dataset/" \
    --checkpoints_dir "$PWD/output/$FILENAME/ckpts/everybodydancenow/" \
    --loadSize 512 \
    --no_instance \
    --resize_or_crop none \
    --no_flip \
    --tf_log \
    --label_nc 6 \
    --niter 4 \
    --niter_decay 1


python "$PWD/everybodydancenow/train_fullts.py" \
--name noam_model_local_non_augmented2 \
--dataroot "$PWD/output/$FILENAME/everybodydancenow/train_dataset/" \
--checkpoints_dir "$PWD/output/$FILENAME/ckpts/everybodydancenow/" \
--load_pretrain "$PWD/output/$FILENAME/ckpts/everybodydancenow/noam_model_global_non_augmented" \
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


# test using test_on_noam_slight_left_ang_right
CUDA_VISIBLE_DEVICES=1 python "$PWD/everybodydancenow/test_fullts.py" \
--name noam_model_local_non_augmented \
--dataroot "$PWD/output/noam_slight_left_ang_right.mp4/everybodydancenow/train_dataset/" \
--checkpoints_dir "$PWD/output/$FILENAME/ckpts/everybodydancenow/" \
--results_dir "$PWD/output/$FILENAME/results/test_on_noam_slight_left_ang_right/" \
--netG local \
--ngf 32 \
--loadSize 512 \
--no_instance \
--resize_or_crop resize \
--how_many 10000 \
--label_nc 6


# test using test_on_noam_slight_left_ang_right
