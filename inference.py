import os
import matplotlib.pyplot as plt
import wandb

for mode in ["Normal", "Augmentation"]:
    for net in ['local', 'global']:
        for test_target in ["noam_front.mp4", "noam_slight_left_ang_right.mp4"]:
            model_name = "noam_model"
            model_name += "_" + net
            model_name += '_augmented' if mode == 'Augmentation' else '_non_augmented'

            options = {
                "model_name": model_name,
                "test_dataset": test_target,
                "train_dataset": "noam_slight_left_ang_right.mp4",
                "local": net == 'local',
                "mode": mode
            }

            print(options)

            if options['test_dataset'] == options['train_dataset']:
                options["target video"] = "Identity"
            elif options['test_dataset'][:4] == options['train_dataset'][:4]:
                options["target video"] = "Simple"
            else:
                options["target video"] = "Others"

            if options["local"]:
                print("local model run")
                flags = {
                    "--name": options["model_name"],
                    "--dataroot" : "output/" + options["test_dataset"] + "/everybodydancenow/train_dataset/",
                    "--checkpoints_dir": "output/" + options["train_dataset"] + "/ckpts/everybodydancenow/",
                    "--results_dir": "output/" + options["train_dataset"] + "/results/wnb/",
                    "--netG": "local",
                    "--loadSize": 1024,
                    "--ngf": 32,
                    "--how_many": 100000,
                    "--label_nc": 6,
                    "--resize_or_crop": "resize",
                    "--no_instance": True,
                    "--no_flip": True
                }
            else:
                print("global model run")
                flags = {
                    "--name": options["model_name"],
                    "--dataroot" : "output/" + options["test_dataset"] + "/everybodydancenow/train_dataset/",
                    "--checkpoints_dir": "output/" + options["train_dataset"] + "/ckpts/everybodydancenow/",
                    "--results_dir": "output/" + options["train_dataset"] + "/results/wnb/",
                    "--netG": "global",
                    "--loadSize": 512,
                    "--how_many": 100000,
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

            # run inference
            command = 'conda activate caroline & CUDA_VISIBLE_DEVICES=1 python "$PWD/everybodydancenow/test_fullts.py" ' + flags_string

            print(command)
            os.system(command)

            path_to_img = lambda i: "output/" + options['train_dataset'] + "/results/wnb/" + options['model_name'] + "/test_latest/images/" + i + "_synthesized_image.png"
            im = lambda i: plt.imread(path_to_img(i))

            # Log the image
            wandb.log({"img": [wandb.Image(im("001"), caption="Frame 1 (synthesized)")]})
            wandb.log({"img": [wandb.Image(im("050"), caption="Frame 50 (synthesized)")]})
            wandb.log({"img": [wandb.Image(im("150"), caption="Frame 150 (synthesized)")]})

            # run ffmpeg:
            command = "ffmpeg -framerate 30 -start_number 0 -y"
            command += ' -i "output/' + options["train_dataset"] + '/results/wnb/' + options["model_name"] + '/test_latest/images/%03d_synthesized_image.png"'
            command += ' "output/' + options["train_dataset"] + '/results/wnb/' + options["model_name"] + '.ogg"'

            print(command)
            os.system(command)

            wandb.log({"video": wandb.Video(
                'output/' + options["train_dataset"] + '/results/wnb/' + options["model_name"] + '.ogg',
                fps=12,
                format="gif"
            )})

            wandb.finish()