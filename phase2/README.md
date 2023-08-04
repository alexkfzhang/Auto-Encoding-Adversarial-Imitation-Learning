# Auto-Encoding Adversarial Imitation Learning

Here is the code for our paper -- "Auto-Encoding Adversarial Imitation Learning".

Our code is based on simple version of OpenAI Baselines (link: https://github.com/andrewliao11/gail-tf).   

## Prerequisites  

The code requires python3 (>=3.5) with the development headers. You'll also need system packages CMake, OpenMPI and zlib. Those can be installed as follows  

### Usage  
    

sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev


To run the code, you need to install OpenAI Gym (link: https://github.com/openai/gym).  

We use the robotics environment in OpenAI Gym, which needs the MuJoCu physics engine (link: http://www.mujoco.org/).   

The experiments were carried out on a 8-GPUs and 40-CPUs server.  


How to run this code, take Walker2d as a example,

Configurations:

cd #this fold

        export ENV_ID="Walker2d-v2"

        export SAMPLE_STOCHASTIC="False"  

        export STOCHASTIC_POLICY="False" 

for non-noisy expert demonstrations setting

        export PICKLE_PATH=data/Walker2d-v2.pkl

for noisy expert demonstrations setting

        export PICKLE_PATH=data/Walker2d-Noisy.pkl



run the experiments:

        python main.py --env_id $ENV_ID --expert_path $PICKLE_PATH



Our AEAIL train the RL agent on Walker2d-v2, Hopper-v2 and Swimmer-v2 without BC pre-training, while on Ant-v2, HalfCheetah-v2 and Humanoid-v2 with BC pre-training 1e4 iterations

Default setting is without BC pre-training, to train the model with BC pre-training pls refer to main.py and set the "--pretrained" parameter to be True or just run:



        python main.py --env_id $ENV_ID --expert_path $PICKLE_PATH --pretrained True --BC_max_iter 10000



Dataset we have provided in this folder

Main Text Experiment Dataset:

data: Walker2d non-noisy and noisy dataset, each with 25 trajectories



