# Image-based Navigation in Real-World Environments via Multiple Mid-level Representations: Fusion Models Benchmark and Efficient Evaluation
This repository hosts the code related to the paper:

Marco Rosano, Antonino Furnari, Luigi Gulino, Corrado Santoro and Giovanni Maria Farinella, "Image-based Navigation in Real-World Environments via Multiple Mid-level Representations: Fusion Models Benchmark and Efficient Evaluation". <b>Submitted</b> to "Robotics and Autonomous Systems" (RAS), 2022.

For more details please see the project web page at [https://iplab.dmi.unict.it/EmbodiedVN](https://iplab.dmi.unict.it/EmbodiedVN/).



## Overview
This code is built on top of the Habitat-api/Habitat-lab project. Please see the [Habitat project page](https://github.com/facebookresearch/habitat-lab) for more details.

This repository provides the following components:

1. The implementation of the proposed tool, integrated with Habitat, to train visual navigation models on synthetic observations and test them on realistic episodes containing real-world images. This allows the estimation of real-world performance, avoiding the physical deployment of the robotic agent;

2. The official PyTorch implementation of the proposed visual navigation models, which follow different strategies to combine a range of visual [mid-level representations](https://github.com/alexsax/midlevel-reps)

3. the synthetic 3D model of the proposed environment, acquired using the Matterport 3D scanner and used to perform the navigation episodes at train and test time;

4. the photorealistic 3D model that contains real-world images of the proposed environment, labeled with their pose (X, Z, Angle). The sparse 3D reconstruction was performed using the [COLMAP Structure from Motion tool](https://colmap.github.io/), to then be aligned with the Matterport virtual 3D map.

5. An integration with [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) to train and evaluate navigation models with Habitat on sim2real adapted images.

6. The checkpoints of the best performing navigation models.



## Installation

### Requirements

* Python >= 3.7, use version 3.7 to avoid possible issues.
* Other requirements will be installed via `pip` in the following steps.

### Steps

0. (Optional) Create an Anaconda environment and install all on it ( `conda create -n fusion-habitat python=3.7; conda activate fusion-habitat` )

1. Install the Habitat simulator following the [official repo instructions](https://github.com/facebookresearch/habitat-sim/) .The development and testing was done on commit `bfbe9fc30a4e0751082824257d7200ad543e4c0e`, installing the simulator "from source", launching the `./build.sh --headless --with-cuda` command ([guide](https://github.com/facebookresearch/habitat-sim/blob/master/BUILD_FROM_SOURCE.md)). Please consider to follow these suggestions if you encounter issues while installing the simulator.

2. Install the customized Habitat-lab (this repo):
	```bash
	git clone https://github.com/rosanom/mid-level-fusion-nav.git
	cd mid-level-fusion-nav/
	pip install -r requirements.txt
	python setup.py develop --all # install habitat and habitat_baselines

	```

3. Download our dataset (journal version) [from here](https://iplab.dmi.unict.it/EmbodiedVN/), and extract it to the repository folder (`mid-level-fusion-nav/`). Inside the `data` folder you should see this structure:
	```bash
	datasets/pointnav/orangedev/v1/...
	real_images/orangedev/...
	scene_datasets/orangedev/...
	orangedev_checkpoints/...

	```

4. (Optional, to check if the software works properly) Download the [test scenes data](http://dl.fbaipublicfiles.com/habitat/habitat-test-scenes.zip) and extract the zip file to the repository folder (`mid-level-fusion-nav/`). To verify that the tool was successfully installed, run  `python examples/benchmark.py` or `python examples/example.py`.



## Data Structure

All data can be found inside the `mid-level-fusion-nav/data/` folder:
* the `datasets/pointnav/orangedev/v1/...` folder contains the generated train and validation navigation episodes files;
* the `real_images/orangedev/...` folder contains the real world images of the proposed environment and the `csv` file with their pose information (obtained with COLMAP);
* the `scene_datasets/orangedev/...` folder contains the 3D mesh of the proposed environment.
* `orangedev_checkpoints/` is the folder where the checkpoints are saved during training. Place the checkpoint file here if you want to restore the training process or evaluate the model. The system will load the most recent checkpoint file.



## Config Files

There are two configuration files:

`habitat_domain_adaptation/configs/tasks/pointnav_orangedev.yaml`

and 

`habitat_domain_adaptation/habitat_baselines/config/pointnav/ddppo_pointnav_orangedev.yaml`.

In the first file you can change the robot's properties, the sensors used by the agent and the dataset used in the experiment. You don't have to modify it.

In the second file you can decide:
1. if evaluate the navigation models using RGB or mid-level representations;
2. the set of mid-level representations to use;
3. the fusion architecture to use;
4. if train or evaluate the models using real images, or using the CycleGAN sim2real adapted observations.
```bash
...
EVAL_W_REAL_IMAGES: True
EVAL_CKPT_PATH_DIR: "data/orangedev_checkpoints/"

SIM_2_REAL: False #use cycleGAN for sim2real image adaptation?

USE_MIDLEVEL_REPRESENTATION: True
MIDLEVEL_PARAMS:
ENCODER: "simple" # "simple", SE_attention, "mid_fusion", ...
FEATURE_TYPE: ["normal"] #["normal", "keypoints3d","curvature", "depth_zbuffer"]
...
```

## CycleGAN Integration (baseline)

In order to use CycleGAN on Habitat for the sim2real domain adaptation during train or evaluation, follow the steps suggested in the [repository of our previous resease](https://github.com/rosanom/habitat-domain-adaptation).



## Train and Evaluation

To train the navigation model using the DD-PPO RL algorithm, run:

`sh habitat_baselines/rl/ddppo/single_node_orangedev.sh`

To evaluate the navigation model using the DD-PPO RL algorithm, run:

`sh habitat_baselines/rl/ddppo/single_node_orangedev_eval.sh`

For more information about DD-PPO RL algorithm, please check out the [habitat-lab dd-ppo repo page](https://github.com/facebookresearch/habitat-lab/tree/master/habitat_baselines/rl/ddppo).



## License
The code in this repository, the 3D models and the images of the proposed environment are MIT licensed. See the [LICENSE file](LICENSE) for details.

The trained models and the task datasets are considered data derived from the correspondent scene datasets.
- Matterport3D based task datasets and trained models are distributed with [Matterport3D Terms of Use](http://kaldir.vc.in.tum.de/matterport/MP_TOS.pdf) and under [CC BY-NC-SA 3.0 US license](https://creativecommons.org/licenses/by-nc-sa/3.0/us/).
- Gibson based task datasets, the code for generating such datasets, and trained models are distributed with [Gibson Terms of Use](https://storage.googleapis.com/gibson_material/Agreement%20GDS%2006-04-18.pdf) and under [CC BY-NC-SA 3.0 US license](https://creativecommons.org/licenses/by-nc-sa/3.0/us/).

<!--
## Citation
If you use the code/data of this repository in your research, please cite the paper:

```
@inproceedings{rosano2020fusion,
  title={Image-based Navigation in Real-World Environments via Multiple Mid-level Representations: Fusion Models Benchmark and Efficient Evaluation},
  author={Rosano, Marco and Furnari, Antonino and Gulino, Luigi and Santoro, Corrado and Farinella, Giovanni Maria},
  booktitle={Robotics and Autonomous Systems (RAS)}
  year={2022}
}
```
-->


## Acknowledgements

This research is supported by [OrangeDev s.r.l](https://www.orangedev.it/), by [Next Vision s.r.l](https://www.nextvisionlab.it/), the project MEGABIT - PIAno di inCEntivi per la RIcerca di Ateneo 2020/2022 (PIACERI) – linea di intervento 2, DMI - University of Catania, and the grant MIUR AIM - Attrazione e Mobilità Internazionale Linea 1 - AIM1893589 - CUP E64118002540007.
