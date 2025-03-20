#from trackeval.metrics import hota
import os
import sys
sys.path.append(os.path.abspath('TrackEval'))
from collections import defaultdict

import argparse
import numpy as np
from scipy.optimize import linear_sum_assignment
from trackeval.metrics._base_metric import _BaseMetric
from trackeval import _timing
from trackeval.metrics.hota import HOTA
from trackeval.metrics.identity import Identity

def calculate_metrics_taking_only_GTobject_into_account(tracker_data, gt_data, iou_threshold=0.01):
    """
    Evaluates metrics using only GT objects. Instead of relying on matching IDs (since
    GT and tracker IDs differ), we filter tracker detections per frame. We retain only
    tracker detections that have an IoU >= iou_threshold with at least one GT box.
    In frames where no valid tracker detections are found, a dummy detection is inserted
    to avoid failure in the HOTA function.
    """
    # Build mapping dict for GT track IDs (from GT detections only)
    unique_gt_ids = set()
    for frame, dets in gt_data.items():
        for det in dets:
            unique_gt_ids.add(det['track_id'])
    unique_gt_ids = sorted(list(unique_gt_ids))
    gt_id_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_gt_ids)}

    # Filter tracker detections per frame using IoU threshold, only for frames present in GT
    filtered_tracker_data = {}
    for frame in gt_data.keys():
        valid_tr_dets = []
        if frame in tracker_data:
            for tr in tracker_data[frame]:
                # Compute IoU with each GT detection in this frame
                ious = [iou(gt['bbox'], tr['bbox']) for gt in gt_data[frame]]
                if ious and max(ious) >= iou_threshold:
                    valid_tr_dets.append(tr)
        # If no valid tracker detection exists, insert the dummy detection
        if not valid_tr_dets:
            valid_tr_dets = [{
                'bbox': [0, 0, 0, 0],
                'category_id': 0,
                'track_id': 0,
                'conf': 0
            }]
        filtered_tracker_data[frame] = valid_tr_dets

    # Build mapping for tracker track IDs from the filtered data
    unique_tracker_ids = set()
    for frame in gt_data.keys():
        for det in filtered_tracker_data[frame]:
            unique_tracker_ids.add(det['track_id'])
    unique_tracker_ids = sorted(list(unique_tracker_ids))
    tracker_id_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_tracker_ids)}

    # Evaluate on frames that exist in GT
    all_frames = sorted(gt_data.keys())
    gt_ids_list = []
    tracker_ids_list = []
    similarity_scores_list = []
    total_tracker_dets = 0
    total_gt_dets = 0

    for frame in all_frames:
        gt_dets = gt_data[frame]
        # Remap GT track IDs
        gt_ids = np.array([gt_id_mapping[det['track_id']] for det in gt_dets])
        tr_dets = filtered_tracker_data.get(frame, [{
            'bbox': [0, 0, 0, 0],
            'category_id': 0,
            'track_id': 0,
            'conf': 0
        }])
        tr_ids = np.array([tracker_id_mapping[det['track_id']] for det in tr_dets])
        # print(gt_ids, tr_ids)
        total_gt_dets += len(gt_dets)
        total_tracker_dets += len(tr_dets)

        # Build similarity matrix using IoU for the current frame
        if len(gt_dets) > 0 and len(tr_dets) > 0:
            sim_matrix = np.zeros((len(gt_dets), len(tr_dets)), dtype=float)
            for i, gt in enumerate(gt_dets):
                for j, tr in enumerate(tr_dets):
                    sim_matrix[i, j] = iou(gt['bbox'], tr['bbox'])
        else:
            sim_matrix = np.zeros((len(gt_dets), len(tr_dets)), dtype=float)

        gt_ids_list.append(gt_ids)
        tracker_ids_list.append(tr_ids)
        similarity_scores_list.append(sim_matrix)

    num_gt_ids = len(unique_gt_ids)
    num_tracker_ids = len(unique_tracker_ids)

    # Create the data dictionary expected by the HOTA evaluation metric
    data = {
        'num_tracker_dets': total_tracker_dets,
        'num_gt_dets': total_gt_dets,
        'num_gt_ids': num_gt_ids,
        'num_tracker_ids': num_tracker_ids,
        'gt_ids': gt_ids_list,
        'tracker_ids': tracker_ids_list,
        'similarity_scores': similarity_scores_list
    }

    hota_metric = HOTA()
    identity_metric = Identity()  # Assuming your Identity metric is defined similarly
    result_hota = hota_metric.eval_sequence(data)
    result_identity = identity_metric.eval_sequence(data)
    return result_hota, result_identity

def parse_tracking_file(filepath):
    """
    Reads the detection text file and returns data grouped by frame.
    Each element in the returned list corresponds to a single frame,
    which itself is a list of dictionaries.
    Each dictionary has keys: 'bbox' -> [left, top, width, height], 'conf' -> conf_value

    :param filepath: Path to the input text file.
    """

    # Using a dictionary to accumulate detections by frame number:
    frames_dict = defaultdict(list)

    with open(filepath, 'r') as f:
        for line in f:
            # Strip and skip any empty lines
            line = line.strip()
            if not line:
                continue

            # Split line into fields
            fields = line.split(',')
            # fields are expected as: frame, track_id, left, top, width, height, conf, -1, -1, -1

            frame = int(fields[0].strip())
            track_id = int(fields[1].strip())
            left  = float(fields[2].strip())
            top   = float(fields[3].strip())
            width = float(fields[4].strip())
            height= float(fields[5].strip())
            conf  = float(fields[6].strip())

            # Construct detection dictionary
            detection = {
                'bbox': [left, top, width, height],
                'conf': conf,
                'track_id': track_id
            }

            # Append the detection to the corresponding frame
            frames_dict[frame].append(detection)

    return frames_dict


def save_tracking_data(filepath, tracking_data):
    """
    Saves tracking data to a file in the format:
      frame,id,left,top,width,height,conf,-1,-1,-1

    :param filepath: Path to the output text file.
    :param tracking_data: A list of frames (list),
                          where each frame is a list of dictionaries.
                          Each dictionary has keys:
                            {
                                'id': <integer ID>,
                                'bbox': [left, top, width, height],
                                'conf': <float confidence>
                            }
    """
    with open(filepath, 'w') as f:
        # 'frame_idx' will start from 1, but adjust if your frames are 0-based
        for frame_idx, detections in enumerate(tracking_data, start=1):
            for det in detections:
                box_id = det['track_id']
                left, top, width, height = det['bbox']
                conf = det['conf']
                # Write one line per detection
                line = f"{frame_idx},{box_id},{left:.2f},{top:.2f},{width:.2f},{height:.2f},{conf:.2f},-1,-1,-1"
                f.write(line + "\n")

def iou(box_a, box_b):
    """
    Computes the Intersection-over-Union (IoU) of two boxes.
    Each box is in the format [x, y, w, h].
    """
    # Convert [x, y, w, h] to (xmin, ymin, xmax, ymax)
    ax1, ay1 = box_a[0], box_a[1]
    ax2, ay2 = ax1 + box_a[2], ay1 + box_a[3]

    bx1, by1 = box_b[0], box_b[1]
    bx2, by2 = bx1 + box_b[2], by1 + box_b[3]

    # Intersection rectangle
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    intersection_area = inter_w * inter_h

    # Areas of each box
    area_a = box_a[2] * box_a[3]  # w*h
    area_b = box_b[2] * box_b[3]

    union_area = area_a + area_b - intersection_area
    if union_area == 0:
        return 0.0
    return intersection_area / union_area


def union_box(box_a, box_b):
    """
    Returns the bounding-box union of the two boxes
    (the minimal rectangle that encloses both).
    Each box is in the format [x, y, w, h].
    """
    ax1, ay1 = box_a[0], box_a[1]
    ax2, ay2 = ax1 + box_a[2], ay1 + box_a[3]

    bx1, by1 = box_b[0], box_b[1]
    bx2, by2 = bx1 + box_b[2], by1 + box_b[3]

    union_x1 = min(ax1, bx1)
    union_y1 = min(ay1, by1)
    union_x2 = max(ax2, bx2)
    union_y2 = max(ay2, by2)

    return [union_x1, union_y1, union_x2 - union_x1, union_y2 - union_y1]


def union_of_boxes(list_of_boxes):
    """
    Given a list of [x, y, w, h] boxes, returns the bounding box that encloses them all.
    """
    if not list_of_boxes:
        return None

    # Initialize x1, y1 with a large value, x2, y2 with a small value
    x1 = float('inf')
    y1 = float('inf')
    x2 = float('-inf')
    y2 = float('-inf')

    for (x, y, w, h) in list_of_boxes:
        # Convert [x, y, w, h] into corners
        bx1, by1 = x, y
        bx2, by2 = x + w, y + h

        # Update union coordinates
        x1 = min(x1, bx1)
        y1 = min(y1, by1)
        x2 = max(x2, bx2)
        y2 = max(y2, by2)

    # Convert corners back to [x, y, w, h]
    return [x1, y1, x2 - x1, y2 - y1]


if __name__ == '__main__':
    hotas = []
    idf1s = []
    camera_names = ['S03_c001', 'S03_c002', 'S03_c003', 'S03_c004', 'S03_c005', 'S03_c006']
    for cam_name in camera_names:
        gt_path = f"/home/maria/mcv-c6-2025-team6/data/aic19-track1-mtmc-train/train/complete_videos/S03/gt/{cam_name}_gt.txt"
        det_path = f"/home/maria/mcv-c6-2025-team6/data/aic19-track1-mtmc-train/train/complete_videos/S03/{cam_name}.txt"
        # tracking = parse_tracking_file(f"{path_to_mtmc_train}/train/{seq}/{cam}/mtsc/{trackname}.txt")#
        tracking = parse_tracking_file(gt_path)#
        # annotations = parse_tracking_file(f"{path_to_mtmc_train}/train/{seq}/{cam}/gt/gt.txt")
        annotations = parse_tracking_file(det_path)

        result_hota, result_identity = calculate_metrics_taking_only_GTobject_into_account(tracking, annotations)
        print(result_hota['HOTA'])

        print(result_identity['IDF1'])
        hota = result_hota['HOTA']
        idf1 = result_identity['IDF1']
        hotas.append(np.mean(hota))
        idf1s.append(idf1)
    with open("results_sort_metric.txt", 'a') as exp:
        exp.write(f"SORT, Cameras: [{camera_names}], HOTA (mean): {np.mean(hotas)}, IDF1: {np.mean(idf1s)}\n")