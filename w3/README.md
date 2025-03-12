# mcv-c6-2025-team6

## Week 3 - Object Tracking with Optical Flow & Multi-target single-camera (MTSC) tracking

### Data
Add the AICity_data folder at the same level of the main.py script. The data should be organized as follows:

```
data
└───AICity_data
    └───train
    |   └───S03
    |       └───c010
    |           └───vdo.avi
    |           └───frames
    |               ├── frame_0001.jpg
    |               ├── frame_0002.jpg
    |               ├── ...
    └───ai_challenge_s03_c010-full_annotation.xml
```
We cannot always process video files directly, so you may need to extract frames from vdo.avi before running object detection.

### Install Dependencies

⚠️ Note: We used separate virtual environments for `PyFlow` and `FlowFormer` to avoid version conflicts. Ensure you activate the correct environment before installing dependencies and running models.
`Perceiver IO` was run in Google Colab, so no local dependencies were required.

To install the required packages, execute the following commands:

#### PyFlow environment
```
pip install -r requirements_pyflow.txt
```
Additionally, clone the [`PyFlow`](https://github.com/pathak22/pyflow.git) repository and install its dependencies.

#### FlowFormer environment
```
pip install -r requirements_flowformer.txt
```
Additionally, clone the [`FlowFormer`](https://github.com/drinkingcoder/FlowFormer-Official.git) repository and install its dependencies.


### Week Structure
This project contains the following main Python scripts or Jupyter notebooks:
- task_1_1_pyflow.py: 
- task_1_2.py:
- task_2_inference.py: 
- task_2_tracking.py:
- task_2_sort.py: 
- task_2_utils.py: This file contains utility functions used in task 2.
- task_2_eval.py: This script implements the HOTA metrics for evaluating tracking performance, based on the repository [TrackEval](https://github.com/JonathonLuiten/TrackEval) and [Team 8's](https://github.com/mcv-m6-video/mcv-c6-2025-team8) contribution.

### Usage
#### Task 1: Optical flow estimation
