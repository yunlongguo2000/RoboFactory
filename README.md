# Multi-Agent Embodied Intelligence Challenge

This repository contains the code for Multi-Agent Control Track of the  [Multi-Agent Embodied Intelligence Challenge](https://mars-eai.github.io/MARS-Challenge-Webpage/)

This track focuses on low-level policy execution in physically realistic simulation environments. It utilizes RoboFactory, a simulation benchmark for embodied agents. Participants are required to deploy and control multiple embodied agents (e.g., robotic arms) to collaboratively complete manipulation-centric tasks like block stacking.

Each task is an episode where agents interact with dynamic objects in a shared workspace under partial observability and randomized conditions. The core challenge lies in achieving robust, learned coordination across multiple agents.

## Update
- **2025.09.01**: Official Round begins — Please re-download the latest asset files. Official tasks: **`four_robots_stack_cube`**, **`place_cube_in_cup`**, **`strike_cube_hard`**, **`three_robots_place_shoes`** 
- **2025.08.18**: Warm-up Round begins  

> **Note**  
> Participants who need to submit model results, please fill out  [this form](https://forms.gle/2dPYe3rKXHc8bk3Y8) to receive submission details.

## 🗓️ Competition Timeline

| Date            | Phase             | Description                                                                 |
|-----------------|-------------------|-----------------------------------------------------------------------------|
| August 18th     | Warmup Round      | Environment opens for teams to explore and get familiar (no prizes).        |
| September 1st   | Official Round    | Competition begins with unseen tasks and prize challenges.                  |
| October 31th  | Official Round Ends | Expected closing of the official round.                                    |
| December        | Award Ceremony    | Final results and awards will be announced.                                 |

## Installation

### 1. Clone the Repository:

First, install [vulkan](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html#vulkan). 
The repository contains submodules, thus please check it out with

```bash
# HTTP
# Use --recurse to clone submodules
git clone https://github.com/MARS-EAI/RoboFactory.git --recurse
```

or

```bash
# SSH
git clone git@github.com:MARS-EAI/RoboFactory.git --recurse
```

### 2. Create a conda environment:

```bash
conda create -n mars python=3.10
conda activate mars
```

### 3. Install dependencies:

```bash
cd RoboFactory/
pip install -r requirements.txt
```

### 4. Download assets:

```bash
python script/download_assets.py 
```

### 5. Check the Environment by Running the following line (need graphical desktop)

```bash
python script/run_task.py configs/table/take_photo.yaml
```

### 🛠 Installing OpenGL/EGL Dependencies on Headless Debian Servers

If you are running simulation environments on a **headless Debian server** without a graphical desktop, you can try to install a minimal set of OpenGL and EGL libraries to ensure compatibility.

Run the following commands to install the necessary runtime libraries:

```bash
sudo apt update
sudo apt install libgl1 libglvnd0 libegl1-mesa libgles2-mesa libopengl0
```
Then you can check the environment by running the code in next section (Data Generation).

## Data Generation (RoboFactory)

**Available Tasks for Warmup Round:**
- `long_pipeline_delivery`
- `place_food`
- `take_photo`
- `three_robots_stack_cube`

**Available Tasks for Official Round:**
- `four_robots_stack_cube`
- `strike_cube_hard`
- `three_robots_place_shoes`
- `place_cube_in_cup`

You can use the following command to run the tasks using the expert policy solution

```bash
python script/run_task.py configs/table/[task name].yaml
```

Use the following Commands to generate datas

```bash
python script/generate_data.py configs/table/[task name].yaml [number of trajectories] --save-video
# Example:  python script/generate_data.py configs/table/place_food.yaml 100 --save-video

# Multi-processing
python script/generate_data.py configs/table/[task name].yaml [number of trajectories]  --save-video --num-procs [number of processes]
```
The generated raw data will be saved by default in the demo/ directory.


## Training the Baseline Policy (Policy-Lightning)

The baseline policy for this project is implemented using the Policy-Lightning framework. 

For now, policy provided is a 2D image-based diffusion policy. This policy is a global policy that takes camera images from all agents and uses a single diffusion model to control all agents jointly. The implementation can be found in the `Policy-Lightning` folder. We plan to support more policies soon.

To learn more about the Policy-Lightning framework, please refer to  [Policy-Lightning/README.md](Policy-Lightning/README.md).



###  1. Data Preparation

First, create a data folder (like data/), then convert the generated raw data into the training format:

```bash 
python Policy-Lightning/script/image/extract.py --dataset_path [path_to_raw_data] --output_path [path_to_output_data] --load_num [number_of_episodes] --agent_num [number_of_agents]

# Example:  python Policy-Lightning/script/image/extract.py --dataset_path demos/PlaceFood-rf/motionplanning/xxx.h5 --output_path data/place_food.h5 --load_num 100 --agent_num 2
```

For a comprehensive explanation of the data format for training, please refer to [docs/data_convert.md](docs/data_convert.md).


### 2. Training
**2D Diffusion Policy:**

```bash
# General format:
python Policy-Lightning/workspace.py --config-name=[policy_config] task=[task_name]

# Example:
python Policy-Lightning/workspace.py --config-name=dp2 task=place_food
```

### 3. Evaluation

You can evaluate your trained checkpoints from policy-lightning using the `eval_policy_lt.py` script provided in the project root. This script allows you to run your policy in the simulation environment and report performance metrics.

Before running the evaluation, make sure to **configure the arguments at the top of `eval_policy_lt.py`** to match your experiment settings (e.g., checkpoint path, number of agents, data location, etc.).


```bash
python eval_policy_lt.py --ckpt_path=[checkpoint path] --config=[task config] --max_steps=[policy try max steps]
```


## Implement Your Own policy

You can implement your own policy under the `custom_policy` directory. This folder contains a `deploy_policy.py` file, which provides an interface for integrating your custom policy with the evaluation and training framework. 

To adapt your policy to the framework, simply wrap your policy logic within the interface provided by `deploy_policy.py`. This ensures compatibility with the rest of the system and allows you to evaluate your policy using the provided scripts (such as `eval_policy.py`).

For example, you can check the `deploy_policy.py` file under the `Policy-Lightning` folder, where there is an example of how to wrap your policy.

After training, you can use the eval_policy.py to eval your custom policy， you should also make sure to **configure the arguments at the top of `eval_policy.py`** to match your experiment settings. 

**Note: Reserve the method "update_obs", "get_action", and "reset" for policy execution.**


## Submission Instructions

We recommend using our `python_submit.py` script for submission.

> **Note:** To ensure fairness and security, you may only submit after completing or deleting your previous submission.
>  **Note:** If the submission task has already been evaluated, you will not be able to upload it again.

---

### Prerequisites

1. Download the Python submission script `python_submit.py` from our GitHub repository.
2. Make sure you are using **Python 3.6 or higher**, and have the **requests** library installed:

```bash
pip install requests
```

1. Ensure your network is able to connect to the competition server.

---

### Prepare Your Submission Files

You need to prepare the following three folders in your current working directory and organize them according to the specific structure:

#### 📁 Directory Structure Overview

```
submission/
├── checkpoints/
│   └── last.ckpt                    # Trained model checkpoint file
├── configs/
│   └── place_food.yaml              # Task configuration file (must match task name)
└── custom_policy/
    ├── deploy_policy.py             # Deployment policy file (required)
    └── [other files...]             # Other related code files and subdirectories
```

------

#### `checkpoints` Folder

Contains a `.ckpt` format model checkpoint file. Filename can be arbitrary but should be meaningful.

**Example path:** `./checkpoints/last.ckpt`

------

#### `configs` Folder

Contains YAML format task configuration file. Filename must exactly match the selected task name:

- Task 1 (**four_robots_stack_cube**): `./configs/four_robots_stack_cube.yaml`
- Task 2 (**strike_cube_hard**): `./configs/strike_cube_hard.yaml`
- Task 3 (**three_robots_place_shoes**): `./configs/three_robots_place_shoes.yaml`
- Task 4 (**place_cube_in_cup**): `./configs/place_cube_in_cup.yaml`



------

#### `custom_policy` Folder

Must contain `deploy_policy.py` as the evaluation system entry point. Can include any number of auxiliary Python files, subdirectories, and modules.

- **Required file:** `./custom_policy/deploy_policy.py`
- **Optional:** Other Python files, subdirectories, and project structures

------

### Submitting with `python_submit.py`

#### Basic Usage

```bash
python python_submit.py <submission_id> <api_key> <checkpoint_file> <config_file> <policy_folder>
```

#### Argument Description

- `submission_id`: Your submission ID (obtain it from the competition platform after creating a new submission)
- `api_key`: Your API key (obtain it from the competition platform after creating a new submission)
- `checkpoint_file`: Path to the checkpoint file
- `config_file`: Path to the config file
- `policy_folder`: Path to the custom policy folder

**Example:**

```bash
python python_submit.py sub123 key123 ./checkpoints/last.ckpt ./configs/place_food.yaml ./custom_policy
```

------

### Execution Steps

When you run the script, it will perform the following steps:

1. **Input Validation**: Check whether all required files and folders exist
2. **Prepare Submission Structure**: Organize files as required by the competition
3. **File Packaging**: Compress all files into a zip archive
4. **Chunked Upload**: Upload the zip file in chunks (supports resume on failure)
5. **Trigger Evaluation**: Notify the server to start evaluating your submission
6. **Clean Up**: Delete any temporary files created during submission

------

### Network Configuration

By default, the script connects to the server at:

```python
# Default backend URL - can be overridden by environment variable
backend_url = os.environ.get('BACKEND_URL', 'https://mygo.iostream.site')
```

You can override the backend server address by setting the `BACKEND_URL` environment variable:

```bash
export BACKEND_URL=https://mygo.iostream.site
python python_submit.py ...
```

Make sure to verify whether `BACKEND_URL` has been modified before submitting.

------

### Error Handling

- If any file is missing, the script will stop and report the error
- If the upload fails due to network issues, it will retry up to **5 times**
- If the submission has already been evaluated, the upload will be skipped
- **Resume is supported**: you can rerun the same command to continue an interrupted upload

------

### Frequently Asked Questions

**Q: What should I do if I get a "File not found" error?**
 A: Check whether the file paths are correct and whether the files actually exist in the specified locations.

**Q: What if the upload is very slow?**
 A: This is normal for large files. They are uploaded in chunks. If the upload is interrupted, simply rerun the command. Uploaded parts will be skipped.

**Q: How do I get my submission_id and api_key?**
 A: These can be obtained from the official competition website or platform.

**Q: Can I submit multiple versions with different submission_ids at the same time?**
 A: No. The script submits one version at a time. Each `submission_id` corresponds to a separate submission.

**Q: I submitted using python_submit.py. How do I submit another one?**
 A: Refresh the competition platform page.


## (Optional) Camera Configuration

### Customizing Agent Camera Positions and Angles

To adjust the position and orientation of agent cameras in your simulation, follow these steps:

1. **Launch the Interactive Environment**

   Run the following command to open the graphical interface for your task:

   ```bash
   python script/run_task.py configs/table/[task name].yaml
   ```

   This will launch a window where you can visualize and interact with the simulation environment.

2. **Review and Adjust Camera Settings**

   - Use the built-in camera widget to inspect the current camera configuration for each agent.
   - You can manually move and rotate the cameras within the interface to achieve your desired viewpoint.

   ![Camera Widget](docs/camera_widget.png)

3. **Copy and Apply Camera Configuration**

   - Once you are satisfied with the camera's position and angle, click the **"Copy Camera Setting"** button in the interface.
   - This will output a camera configuration snippet similar to:

     ```
     camera = add_camera(name="", width=1920, height=1080, fovy=1.57, near=0.1, far=1e+03)
     camera.set_local_pose(Pose([0.345104, 0.239462, 0.240109], [0.34335, 0.0538675, 0.0197297, -0.937454]))
     ```

   - Extract the pose parameters (the two lists inside `Pose([...], [...])`) and update the corresponding camera entry in your task config file (`configs/table/[task name].yaml`).
   - **Important:** Set the `type` field to `"pose"` in the YAML configuration.

   **Example YAML camera configuration:**

   ```yaml
   - uid: head_camera_agent1
     pose:
       type: pose
       params: [[0.345104, 0.239462, 0.240109], [0.34335, 0.0538675, 0.0197297, -0.937454]]
     width: 320
     height: 240
     fov: 1.5707963268
     near: 0.1
     far: 10
   ```


   Repeat this process for each camera you wish to customize.

**If you are using the baseline policy through policy lightning**, please remember to modify the task config of policy-lightning through `Policy-Lightning\config\task` to match up with your camera shape.

During evaluation for the contest, your camer configuration will be **extracted and be used to evaluate your policy**.

## (Optional) Customizing Data Generation

Episode generation is driven by expert rule-based policies—Python scripts that specify sequences of actions to solve each task. The default expert policies are located in the `planner/solutions` directory. Each `.py` file in the folder implements an expert policy as a solution for a task.

You are encouraged to modify and improve these scripts, to generate your own data that best suits your approach.

You can use the following command to run the tasks using the expert policy solution

```bash
python script/run_task.py configs/table/[task name].yaml
```

## Contact

If you have any questions, feel free to email us on <marschallenge2025@gmail.com>.
