#from trackeval.metrics import hota
import os
import sys
sys.path.append(os.path.abspath('TrackEval'))

import argparse
import numpy as np
from scipy.optimize import linear_sum_assignment
from trackeval.metrics._base_metric import _BaseMetric
from trackeval import _timing
from trackeval.metrics.hota import HOTA
from trackeval.metrics.identity import Identity

from trackeval import utils

def load_mot_txt(file_path, min_frame=None):
    """Loads a MOT-style txt file into a dictionary format for TrackEval."""
    data = {}
    with open(file_path, 'r') as f:
        for line in f:
            frame, track_id, x, y, w, h, conf, _, _, _ = map(float, line.strip().split(','))
            frame = int(frame)
            track_id = int(track_id)
            
            # Convert width and height to positive values
            w = abs(w)
            h = abs(h)

            # Ignore frames before the first GT frame
            if min_frame is not None and frame < min_frame:
                continue
            
            if frame not in data:
                data[frame] = []
            data[frame].append((track_id, x, y, w, h, conf))
    return data

def compute_similarity(gt_data, det_data):
    """Computes similarity scores between ground truth and detections based on IoU."""
    similarity_scores = []
    gt_ids_list = []
    det_ids_list = []
    
    frames = sorted(set(gt_data.keys()).intersection(set(det_data.keys())))
    for frame in frames:
        gt_boxes = [box[1:5] for box in gt_data[frame]]
        det_boxes = [box[1:5] for box in det_data[frame]]
        
        if not gt_boxes or not det_boxes:
            similarity_scores.append(np.zeros((len(gt_boxes), len(det_boxes))))
            gt_ids_list.append(np.array([]))
            det_ids_list.append(np.array([]))
            continue
        
        iou_matrix = np.zeros((len(gt_boxes), len(det_boxes)))
        for i, gt_box in enumerate(gt_boxes):
            for j, det_box in enumerate(det_boxes):
                iou_matrix[i, j] = compute_iou(gt_box, det_box)
        
        similarity_scores.append(iou_matrix)
        gt_ids_list.append(np.arange(len(gt_boxes)))
        det_ids_list.append(np.arange(len(det_boxes)))
    
    return gt_ids_list, det_ids_list, similarity_scores

def compute_iou(box1, box2):
    """Computes IoU between two bounding boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)
    
    inter_area = max(0, xb - xa) * max(0, yb - ya)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0

def get_hota_idf1(gt_path, det_path, output_dir=None):
    """Main function to compute HOTA."""
    gt_data = load_mot_txt(gt_path)
    
    # Get the first GT frame number
    min_frame = min(gt_data.keys()) if gt_data else None
    
    det_data = load_mot_txt(det_path, min_frame)
    
    gt_ids, det_ids, similarity_scores = compute_similarity(gt_data, det_data)
    
    data = {
        'gt_ids': gt_ids,
        'tracker_ids': det_ids,
        'similarity_scores': similarity_scores,
        'num_gt_dets': sum(len(ids) for ids in gt_ids),
        'num_tracker_dets': sum(len(ids) for ids in det_ids),
        'num_gt_ids': len(set(id for ids in gt_ids for id in ids)),
        'num_tracker_ids': len(set(id for ids in det_ids for id in ids)),
    }
    
    hota_metric = HOTA()
    results_hota = hota_metric.eval_sequence(data)
    print("\nHOTA Score:", results_hota['HOTA'])
    
    identity_metric = Identity()
    results_idf1 = identity_metric.eval_sequence(data)
    print("Identity Score:", results_idf1['IDF1'])
    
    if output_dir is not None:
        with open(f"{output_dir}/metrics.txt", 'w') as file:
            line = f"\nHOTA Score: {results_hota['HOTA']}"
            file.write(line)
            line = f"\nIdentity Score: {results_idf1['IDF1']}"
            file.write(line)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process input and output files.")
    parser.add_argument('--gt', type=str, help='Path to the gt file')
    parser.add_argument('--predictions', type=str, help='Path to the predictions file')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    args = parser.parse_args()

    get_hota_idf1(gt_path=args.gt, det_path=args.predictions, output_dir=args.output_dir)