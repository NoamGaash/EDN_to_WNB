import os
import matplotlib.pyplot as plt
import wandb

train_dataset = "noam_slight_left_ang_right.mp4"


for mode in ["Normal", "Augmentation"]:
    for net in ['local', 'global']:
        for test_target in ["dor_front.mp4", "noam_front.mp4", "noam_slight_left_ang_right.mp4"]:
            for ep in ["15", "5"]:
                model_name = "noam_model"
                model_name += "_" + net
                model_name += '_augmented' if mode == 'Augmentation' else '_non_augmented'
                if ep != "5":
                    model_name += '_' + ep
                    if not os.path.isdir("output/" + train_dataset + "/ckpts/everybodydancenow/" + model_name):
                        continue


                options = {
                    "model_name": model_name,
                    "test_dataset": test_target,
                    "train_dataset": train_dataset,
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
                        "--loadSize": 512,
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
                command = 'conda init bash & conda activate caroline & python "$PWD/everybodydancenow/test_fullts.py" ' + flags_string

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

                """wandb.log({"video": wandb.Video(
                    'output/' + options["train_dataset"] + '/results/wnb/' + options["model_name"] + '.ogg',
                    fps=12,
                    format="gif"
                )})"""


                # combine three videos:

                command = "ffmpeg -y \
                    -i  \"output/" + options["train_dataset"] + "/results/wnb/" + options["model_name"] + ".ogg\" \
                    -i \"videos/" + options["test_dataset"] + "\" \
                    -filter_complex \
                    \"[0][1]scale2ref='oh*mdar':'if(lt(main_h,ih),ih,main_h)'[0s][1s]; \
                        [1s][0s]scale2ref='oh*mdar':'if(lt(main_h,ih),ih,main_h)'[1s][0s]; \
                        [0s][1s]hstack,setsar=1\" \
                        \"output/" + options["train_dataset"] + "/results/wnb/" + options["model_name"] + "_combined.mp4\" \
                        -y"

                print(command)
                os.system(command)

                wandb.log({"video": wandb.Video(
                    "output/" + options["train_dataset"] + "/results/wnb/" + options["model_name"] + "_combined.mp4",
                    fps=12,
                    format="gif"
                )})

                wandb.finish()