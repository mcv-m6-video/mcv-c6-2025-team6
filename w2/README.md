# mcv-c6-2025-team6

## Week 2 - Object detection & Tracking

### Data
Add the AICity_data folder at the same level of the main.py script. The data should be organized as follows:

```
data
│
└───AICity_data
    |
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
To install the required packages to run the program, execute the following command:

```bash
pip install -r requirements.txt
```
Additionally, clone the `YOLOv3` repository and install its dependencies:
```
git clone https://github.com/ultralytics/yolov3
cd yolov3
pip install -qr requirements.txt comet_ml
```
### Week Structure
This project contains the following main Python scripts or Jupyter notebooks:
- task1_1_and_1_2_yolo.ipynb: contains the development of tasks 1.1 and 1.2 implemented using `YOLOv3` and `YOLOv5x6`.
- task1_1_and_1_2_fasrcnn.ipynb: contains the development of tasks 1.1 and 1.2 implemented using `Detectron2` for `Faster R-CNN`.
- task1_3.py:
- utils.py: This file contains utility functions used by 1.1, 1.2 and 1.3.
- task2_1.py:
- task2_3.py:

### Usage
#### Task 1: Object detection
##### `YOLO` 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mcv-c6-2025-team6/week2/blob/main/1_1_and_1_2_yolo.ipynb)

To run Task 1 and Task 2 for `YOLO`, you need to set specific parameters depending on the method you want to use (`YOLOv3` or `YOLOv5x6`). Below is a guide on which parameters to modify and the best configurations for each approach.
###### Task 1.1: 'Off-the-shelf' model
Run detection on the video file using the following command, where `MODEL.pt` should be `yolov3.pt` or `yolov5x6.pt`:
```
python yolov3/detect.py --weights MODEL.pt --img 1280 --conf 0.4 --source data/AICity_data/train/S03/c010/vdo.avi --save-txt --class 2 --save-conf
```
This will detect objects in vdo.avi and save results to `yolov3/runs/detect/exp/`.

Run evaluation on the extracted frames using the following command, where `MODEL.pt` should be `yolov3.pt` or `yolov5x6.pt`:
```
python yolov3/val.py --weights MODEL.pt --data yolov3/data/AICity.yaml --img 1280 --task val
```
This will detect objects in each frame and save results to `yolov3/runs/detect/exp/`.
To evaluate YOLO models on the dataset, ensure the dataset is properly formatted and create the file `yolov3/data/AICity.yaml` to specify the dataset paths:
```
train: data/AICity_data/train/
val: data/AICity_data/val/
nc: 1  # Number of classes
names: ['car']
```
###### Task 1.2: Fine-tuning
Then, run training using the following command, where `MODEL.pt` should be `yolov3.pt` or `yolov5x6.pt`:
```
python yolov3/train.py --img 1280 --batch 4 --epochs 20 --data yolov3/data/AICity.yaml --weights MODEL.pt
```
This will fine-tune the selected model using the AICity dataset.
##### `Faster R-CNN` (`Detectron2`) 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mcv-c6-2025-team6/week2/blob/main/task1_1_and_1_2_fasrcnn.ipynb)
###### Task 1.1 & 1.2
The Faster R-CNN implementation is provided in a Jupyter Notebook designed to run in `Google Colab`. The notebook is structured into sections corresponding to:
- Task 1.1: Running the model off-the-shelf.
- Task 1.2: Fine-tuning the model on the AICity dataset.
To use it, open the notebook in `Google Colab` and follow the provided instructions.
#### Task 2: Object tracking
