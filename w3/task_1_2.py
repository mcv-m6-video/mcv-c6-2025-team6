import os
import cv2
import argparse
import numpy as np
from collections import defaultdict
from task_1_1_pyflow import compute_flow
from tqdm import tqdm

def get_next_points(flow, prev_pts):
    '''Compute the next positions of tracked points using optical flow.

    Args:
        flow (numpy.ndarray): Optical flow field (H, W, 2) where each pixel contains (u, v) displacement vectors.
        prev_pts (numpy.ndarray): Array of previous points with shape (N, 1, 2), where each row contains (x, y) coordinates.

    Returns:
        numpy.ndarray: Updated positions of the tracked points after applying the optical flow.
    '''
    flow_u, flow_v = flow[:,:,0], flow[:,:,1]
    
    next_pts = prev_pts.copy()
    for i, (x, y) in enumerate(prev_pts.squeeze()):
        x = 1919 if x >= 1920 else x
        y = 1079 if y >= 1080 else y
        u = flow_u[int(y), int(x)]
        v = flow_v[int(y), int(x)]
        next_pts[i] = [x + u, y + v]

    return next_pts

def compute_optical_flow(prev_frame, next_frame, model="pyflow", output_dir=None, viz=False):
    '''Compute optical flow between two consecutive frames.

    Args:
        prev_frame (numpy.ndarray): Grayscale or color image representing the previous frame.
        next_frame (numpy.ndarray): Grayscale or color image representing the next frame.
        model (str, optional): Optical flow model to use ('pyflow', 'flowformer', or 'perceiveio'). Defaults to "pyflow".
        output_dir (str, optional): Directory where computed optical flow should be saved. Defaults to None.
        viz (bool, optional): Whether to enable visualization of the computed flow. Defaults to False.

    Returns:
        numpy.ndarray: Computed optical flow field (H, W, 2), where each pixel contains (u, v) displacement vectors.
    '''
    flow = compute_flow(model, prev_frame, next_frame, output_dir=output_dir, viz=viz)
    return flow

def iou(boxA, boxB):
    '''Compute the Intersection over Union (IoU) between two bounding boxes.

    Args:
        boxA (tuple): Bounding box in the format (x_min, y_min, x_max, y_max).
        boxB (tuple): Bounding box in the format (x_min, y_min, x_max, y_max).

    Returns:
        float: IoU value, ranging from 0 (no overlap) to 1 (perfect overlap).
    '''
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def track_objects(frames, detections, output_dir=None, viz=False, force_compute_flow=False):
    '''Track objects across multiple frames using optical flow.

    Args:
        frames (list): List of file paths to frame images.
        detections (list): List of detections per frame. Each frame's detections are a list of bounding boxes (x_min, y_min, x_max, y_max).
        output_dir (str, optional): Directory where tracking results should be saved. Defaults to None.
        viz (bool, optional): Whether to enable visualization. Defaults to False.
        force_compute_flow (bool, optional): Whether to recompute optical flow even if cached results exist. Defaults to False.

    Returns:
        dict: Dictionary mapping object IDs to lists of tracked bounding boxes across frames.
    '''
    tracked_objects = {}
    object_id = 0

    # Load the first frame
    prev_frame = cv2.imread(frames[0])
    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    with open("tracked_objects_log.txt", "w") as f:
        # Initialize object IDs in the first frame
        for det in detections[0]:
            tracked_objects[object_id] = [det]
            frame_id = int(os.path.splitext(os.path.basename(frames[0]))[0])
            xtl, ytl, xbr, ybr = det
            f.write(f"{frame_id}, {object_id}, {xtl}, {ytl}, {xbr-xtl}, {ybr-ytl}\n")
            object_id += 1
        
        for i in tqdm(range(1, len(frames)), desc="Tracking objects"):
            next_frame = cv2.imread(frames[i])
            frame_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
            prev_boxes = [obj[-1] for obj in tracked_objects.values()]
            
            if not prev_boxes:
                continue
            
            prev_pts = np.array([[(b[0]+b[2])//2, (b[1]+b[3])//2] for b in prev_boxes], dtype=np.float32)

            output_path_frame = os.path.join(output_dir, os.path.basename(frames[i-1]))
            flow_file = output_path_frame.split('.')[0] + "_flow_pred.npy"
            # print(flow_file)
            if os.path.exists(flow_file) and not force_compute_flow:
                flow_pred = np.load(flow_file)
            else:
                flow_pred = compute_optical_flow(prev_frame_gray, frame_gray, output_dir=output_path_frame, viz=viz)
            
            next_pts = get_next_points(flow_pred, prev_pts)
            for j, _detection in enumerate(detections[i]):
                best_iou = 0
                best_id = None
        
                det_x_center = (_detection[0] + _detection[2]) // 2
                det_y_center = (_detection[1] + _detection[3]) // 2
                detection = [det_x_center - 10, det_y_center - 10, det_x_center + 10, det_y_center + 10]

                for k, prev_box in enumerate(prev_boxes):
                    predicted_box = [int(next_pts[k][0]-10), int(next_pts[k][1]-10), int(next_pts[k][0]+10), int(next_pts[k][1]+10)]
                    current_iou = iou(detection, predicted_box)
                    if current_iou > best_iou and current_iou > 0.5:
                        best_iou = current_iou
                        best_id = list(tracked_objects.keys())[k]
                
                frame_id = int(os.path.splitext(os.path.basename(frames[i]))[0])
                if best_id is not None:
                    tracked_objects[best_id].append(detection)
                    xtl, ytl, xbr, ybr = detection
                    f.write(f"{frame_id}, {best_id}, {xtl}, {ytl}, {xbr-xtl}, {ybr-ytl}\n")
                else:
                    tracked_objects[object_id] = [detection]
                    xtl, ytl, xbr, ybr = detection
                    f.write(f"{frame_id}, {object_id}, {xtl}, {ytl}, {xbr-xtl}, {ybr-ytl}\n")
                    object_id += 1
            prev_frame_gray = frame_gray.copy()
    
    return tracked_objects

def load_frames_from_folder(folder):
    '''Load frame image file paths from a given folder.

    Args:
        folder (str): Path to the folder containing image frames.

    Returns:
        list: Sorted list of image file paths.
    '''
    frame_files = sorted(os.listdir(folder))
    frames = [os.path.join(folder, file) for file in frame_files]
    print(f"Loaded {len(frames)} frames.")
    return frames

def load_detections_from_txt(file_path):
    '''Load object detections from a text file.

    Args:
        file_path (str): Path to the text file containing detections.

    Returns:
        list: List of detections per frame, where each frame's detections are a list of bounding boxes (x_min, y_min, x_max, y_max).
    '''
    detections = {}
    with open(file_path, 'r') as f:
        for line in f:
            frame_id, x_min, y_min, w, h = map(int, line.strip().split(','))
            x_max, y_max = x_min + w, y_min + h
            if frame_id not in detections:
                detections[frame_id] = []
            detections[frame_id].append((x_min, y_min, x_max, y_max))
    
    max_frame = max(detections.keys())
    detection_list = [detections.get(i, []) for i in range(1, max_frame + 1)]
    print(f"Loaded detections for {len(detections)} frames.")
    return detection_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object tracking using optical flow models. This script processes a sequence of frames and detections to track objects across frames.")
    
    parser.add_argument("-model", dest="model", type=str, default="flowformer", choices=["pyflow", "flowformer", "perceiveio"], 
                        help="Choose the optical flow model: 'pyflow', 'perceiveio', or 'flowformer'.")
    
    parser.add_argument("-force_compute", dest="force_compute", action="store_true", 
                        help="Force optical flow computation (ignore existing flow file).")
    
    parser.add_argument("-frames", dest="frames", type=str, default="FlowFormer-Official/datasets/KITTI/testing", 
                        help="Path to the folder containing the sequence of frames.")
    
    parser.add_argument("-detections", dest="detections", type=str, default="data/detections.txt", 
                        help="Path to the text file containing object detections for each frame.")
    
    parser.add_argument("-output_dir", dest="output_dir", type=str, default="results/data", 
                        help="Path to the directory where the tracking results will be saved.")

    args = parser.parse_args()
    
    # Example usage (assuming frames and detections are available)
    frames = load_frames_from_folder(args.frames) # frames = ['frame1.jpg', 'frame2.jpg', ...]
    detections = load_detections_from_txt(args.detections) # detections = [[(x1, y1, x2, y2), (x1, y1, x2, y2)], ...]
    tracked = track_objects(frames, detections, output_dir=os.path.join(args.output_dir, args.model), viz=True)
    print(tracked)