import os
import matplotlib.pyplot as plt
import wandb

options = {
    "model_name": "noam_model_local_augmented",
    "test_dataset": "noam_slight_left_ang_right.mp4",
    "train_dataset": "noam_slight_left_ang_right.mp4",
    "local": True
}

if options["local"]:
    flags = {
        "--name": options["model_name"],
        "--dataroot" : "output/" + options["test_dataset"] + "/everybodydancenow/train_dataset/",
        "--checkpoints_dir": "output/" + options["train_dataset"] + "/ckpts/everybodydancenow/",
        "--results_dir": "output/" + options["train_dataset"] + "/results/wnb/",
        "--netG": "local",
        "--loadSize": 1024,
        "--ngf": 32,
        "--how_many": 5,
        "--label_nc": 6,
        "--resize_or_crop": "resize",
        "--no_instance": True,
        "--no_flip": True
    }
else:
    flags = {
        "--name": options["model_name"],
        "--dataroot" : "output/" + options["test_dataset"] + "/everybodydancenow/train_dataset/",
        "--checkpoints_dir": "output/" + options["train_dataset"] + "/ckpts/everybodydancenow/",
        "--results_dir": "output/" + options["train_dataset"] + "/results/wnb/",
        "--netG": "global",
        "--loadSize": 512,
        "--how_many": 5,
        "--label_nc": 6,
        "--ngf": 64,
        "--resize_or_crop": "resize_and_crop",
        "--no_instance": True,
        "--no_flip": True
    }

options.update(flags)


wandb.init(project="test-drive", config=options)

def flag_to_string(flag):
    k, v = flag
    if v is True:
        return k
    if v is False:
        return ""
    return k + " " + str(v)

flags_string = ' '.join([flag_to_string(flag) for flag in flags.items()])

command = 'conda activate caroline & CUDA_VISIBLE_DEVICES=1 python "$PWD/everybodydancenow/test_fullts.py" ' + flags_string

print(command)
os.system(command)

path_to_img = "output/" + options['train_dataset'] + "/results/wnb/" + options['model_name'] + "/test_latest/images/001_synthesized_image.png"
im = plt.imread(path_to_img)

# Log the image
wandb.log({"img": [wandb.Image(im, caption="Step 1 (synthesized)")]})

wandb.finish()