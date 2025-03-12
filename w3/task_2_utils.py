import json
import random
import numpy as np

def generate_random_color_for_object():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
def calculate_euclidean_distance_between_points(p1, p2):
    return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def calculate_iou_between_boxes(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    xx1 = max(x1_1, x1_2)
    yy1 = max(y1_1, y1_2)
    xx2 = min(x2_1, x2_2)
    yy2 = min(y2_1, y2_2)

    w = max(0, xx2 - xx1)
    h = max(0, yy2 - yy1)
    intersection_area = w * h

    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1) 
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)  
    union_area = box1_area + box2_area - intersection_area

    iou = intersection_area / union_area if union_area > 0 else 0
    return iou

def filter_overlapping_boxes_based_on_iou(boxes, iou_threshold=0.9):
    filtered_boxes = []
    for i, box1 in enumerate(boxes):
        keep = True
        for j, box2 in enumerate(filtered_boxes):
            if calculate_iou_between_boxes(box1, box2) > iou_threshold:
                keep = False
                break
        if keep:
            filtered_boxes.append(box1)
    return filtered_boxes

def read_bounding_boxes_from_file(txt_file):
    frame_data = {}
    with open(txt_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            frame_num = int(parts[0])
            x1, y1, x2, y2 = map(int, parts[1:])
            if frame_num not in frame_data:
                frame_data[frame_num] = []
            frame_data[frame_num].append([x1, y1, x2, y2])
    return frame_data