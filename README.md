# Dexterity from Touch: Self-Supervised Pre-Training of Tactile Representations with Robotic Play
[[Paper]](TODO) [[Project Website]](https://tactile-dexterity.github.io/) [[Data]](TODO)

**Authors**: [Irmak Guzey](https://irmakguzey.github.io/), [Ben Evans](https://bennevans.github.io/), [Soumith Chintala](https://soumith.ch/) and [Lerrel Pinto](https://www.lerrelpinto.com/), New York University and Meta AI


This repository includes the official implementation of [T-Dex](https://tactile-dexterity.github.io/), including the training pipeline of tactile encoders and the real-world deployment of the non-parametric imitation learning policies for dexterous manipulation tasks using [Allegro hand](https://www.wonikrobotics.com/research-robot-hand) with [XELA sensors integration](https://xelarobotics.com/en/integrations) and [Kinova arm](https://assistive.kinovarobotics.com/product/jaco-robotic-arm). 

Datasets for the play data and the demonstrations is uploaded in [this Google Drive link](TODO).

## Robot Runs
<p align="center">
  <img width="30%" src="https://github.com/tactile-dexterity/tactile-dexterity.github.io/blob/main/mfiles/main_task/gamepad.gif">
  <img width="30%" src="https://github.com/tactile-dexterity/tactile-dexterity.github.io/blob/main/mfiles/main_task/bottle_opening.gif">
  <img width="30%" src="https://github.com/tactile-dexterity/tactile-dexterity.github.io/blob/main/mfiles/main_task/book_opening.gif">
 </p>

 <p align="center">
  <img width="30%" src="https://github.com/holo-dex/holo-dex.github.io/blob/website/mfiles/main_task/bottle-opening.gif">
  <img width="30%" src="https://github.com/holo-dex/holo-dex.github.io/blob/website/mfiles/main_task/card-sliding.gif">
 </p>

 ## Method
![Holo-Dex](https://github.com/holo-dex/holo-dex.github.io/blob/website/mfiles/Intro.png)
Holo-Dex consists of two phases: demonstration colleciton, which is performed in real-time with visual feedback to VR Headset, and demonstration-based policy learning, which can learn to solve dexterous tasks from a limited number of demonstrations.

## Pipeline Installation and Setup
The pipeline requires [ROS](http://wiki.ros.org/noetic/Installation/Ubuntu) for Server-Robot communication. This Package uses the Allegro Hand and Kinova Arm controllers from [DIME-Controllers](https://github.com/NYU-robot-learning/DIME-Controllers). This implementation uses Realsense cameras and which require the [`librealsense`](https://github.com/IntelRealSense/librealsense#installation-guide) API. Also, [OpenCV](https://pypi.org/project/opencv-python/) is required for image compression and other purposes.

After installing all the prerequisites, you can install this pipeline as a package with pip:
```
pip install -e .
```

You can test if it has installed correctly by running `import holodex` from a python shell.

Install the VR Application in your Oculus Headset using the APK provided [here](https://github.com/SridharPandian/Holo-Dex/releases/tag/VR). To setup the VR Application in your Oculus Headset and enter the robot server's IP address (should be in the same network). The following stream border color codes indicate the following:
- Green - the right hand keypoints are being streamed
- Blue - the left hand keypoints are being stream. 
- Red - the stream is paused and gives access to the menu.

<p align="center">
  <img width="70%" src="https://github.com/holo-dex/holo-dex.github.io/blob/website/mfiles/color-code-indicator.gif">
</p>

## Running the teleop
To use the Holo-Dex teleop module, open the VR Application in your Oculus Headset. On the robot server side, start the [controllers](https://github.com/NYU-robot-learning/DIME-Controllers) first followed by the following command to start the teleop:
```
python teleop.py
```
The Holo-Dex teleop configurations can be adjusted in `configs/tracker/oculus.yaml`. The robot camera configurations can be adjusted in `configs/robot_camera.yaml`.

The package also contains an 30 Hz teleop implementation of [DIME](https://arxiv.org/abs/2203.13251) and you can run it using the following command:
```
python teleop.py tracker=mediapipe tracker/cam_serial_num=<realsense_camera_serial_number>
``` 

## Data
All our data can be found in this URL: [https://drive.google.com/drive/folders/1PiuqYkG7O1sIxE7YewVkni6ohLuNY7vF?usp=sharing](https://drive.google.com/drive/folders/1PiuqYkG7O1sIxE7YewVkni6ohLuNY7vF?usp=sharing)

To collect demonstrations using this framework, run the following command:
```
python record_data.py -n <demonstration_number>
```

To filter and process data from the raw demonstrations run the following command:
```
python extract_data.py
```
You can change the data extraction configurations in `configs/demo_extract.yaml`.

## Training Neural Networks
You can train encoders using Self-Supervised methods such as [BYOL](https://arxiv.org/abs/2006.07733), [VICReg](https://arxiv.org/abs/2105.04906), [SimCLR](https://arxiv.org/abs/2002.05709) and [MoCo](https://arxiv.org/abs/2104.02057). Use the following command to train a resnet encoder using the above mentioned SSL methods:
```
python train_ssl.py ssl_method=<byol|vicreg|simclr|mocov3>
```
The training configurations can be changed in `configs/train_ssl.yaml`. 

You can also train:
- Behavior Cloning:
```
python train_bc.py encoder_gradient=true
```
- [Behavior Cloning-Rep](https://arxiv.org/abs/2008.04899):
```
python train_bc.py encoder_gradient=false
```
The training configurations can be changed in `configs/train_bc.yaml`.

## Deploying Models
This implementation can deploy BC, BC-Rep and INN (all visual) on the robot. To deploy BC or BC-Rep, use the following command:
```
python deploy.py model=BC task/bc.model_weights=<bc_model-weights>
```

To deploy INN use the following command:
```
python deploy.py model=VINN task/vinn.encoder_weights_path=<ssl_encoder_weights_path>
```

You can set a control loop instead of pressing the `Enter` key to get actions using the following command:
```
python deploy.py model=<BC|VINN> run_loop=true loop_frequency=<action_loop_frequency>
```




## Getting started
The following assumes our current working directory is the root folder of this project repository; tested on Ubuntu 20.04 LTS (amd64).

### Setting up the project environments
- Install the project environment:
  ```
  conda env create --file=conda_env.yml
  ```
  This will create a conda environment with the name `tactile_learning`. 
- Activate the environment:
  ```
  conda activate tactile_learning
  ```
- Install the `tactile-learning` package by using `setup.py`.
  ```
  pip install -i .
  ```

### Setting up the project environments
- Install the project environment:
  ```
  conda env create --file=conda_env.yml
  ```
- Activate the environment:
  ```
  conda activate cbet
  ```
- Clone the Relay Policy Learning repo:
  ```
  git clone https://github.com/google-research/relay-policy-learning
  ```
- Install MuJoCo 2.1.0: https://github.com/openai/mujoco-py#install-mujoco
- Install CARLA server 0.9.13: https://carla.readthedocs.io/en/0.9.13/start_quickstart/#a-debian-carla-installation
- To enable logging, log in with a `wandb` account:
  ```
  wandb login
  ```
  Alternatively, to disable logging altogether, set the environment variable `WANDB_MODE`:
  ```
  export WANDB_MODE=disabled
  ```

### Getting the training datasets
Datasets used for training will be uploaded to [this OSF link](https://osf.io/q3dx2).
- Download and unzip the datasets.
- In `./config/env_vars/env_vars.yaml`, set the dataset paths to the unzipped directories.
  - `carla_multipath_town04_merge`: CARLA environment
  - `relay_kitchen`: Franka kitchen environment
  - `multimodal_push_fixed_target`: Block push environment

## Reproducing experiments
The following assumes our current working directory is the root folder of this project repository.

To reproduce the experiment results, the overall steps are:
1. Activate the conda environment with
   ```
   conda activate cbet
   ```
2. Train with `python3 train.py`. A model snapshot will be saved to `./exp_local/...`;
3. In the corresponding environment config, set the `load_dir` to the absolute path of the snapshot directory above;
4. Eval with `python3 run_on_env.py`.

See below for detailed steps for each environment.


## Citation

If you use this repo in your research, please consider citing the paper as follows:
```
@article{arunachalam2022holodex,
  title={Holo-Dex: Teaching Dexterity with Immersive Mixed Reality},
  author={Sridhar Pandian Arunachalam and Irmak Guzey and Soumith Chintala and Lerrel Pinto},
  journal={arXiv preprint arXiv:2210.06463},
  year={2022}
}