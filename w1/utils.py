import cv2
import json
import os
import numpy as np
import xml.etree.ElementTree as ET
import itertools

def extract_frames(video_path, output_folder, fps=10):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(original_fps / fps)  

    frame_count = 0
    saved_count = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break  
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Save frames in '{output_folder}': {saved_count}")


def read_ground_truth(xml_file, classes, n_frames):
    '''
    Extracted from Team3-2024 (https://github.com/mcv-m6-video/mcv-c6-2024-team3/blob/main/Week1/task1_2.py).
    [
        [[x1, y1, x2, y2], [x1, y1, x2, y2], …],
        [[x1, y1, x2, y2], [x1, y1, x2, y2], …], 
        …,
        [[...]]
    ]
    '''
    tree = ET.parse(xml_file)
    root = tree.getroot()

    bbox_info = [[] for _ in range(n_frames)]

    # print(len(bbox_info))

    for track in root.findall('./track'):
        label = track.attrib['label']

        if label in classes:
            for box in track.findall('box'):
                parked = False
                for attribute in box.findall('attribute'):
                    if attribute.attrib.get('name') == 'parked' and attribute.text == 'true':
                        parked = True
                        break
                if not parked:
                    frame = int(box.attrib['frame'])
                    xtl = float(box.attrib['xtl'])
                    ytl = float(box.attrib['ytl'])
                    xbr = float(box.attrib['xbr'])
                    ybr = float(box.attrib['ybr'])
                    
                    bbox_info[frame].append([
                        xtl, ytl, xbr, ybr
                    ])
        
    return bbox_info


def compute_confidence_bbox(bbox, img_shape):
    x1, y1, x2, y2 = bbox
    area = (x2 - x1) * (y2 - y1)
    max_area = img_shape[0] * img_shape[1]  # Área total de la imagen
    return min(1.0, area / max_area)  # Normaliza entre 0 y 1

def compute_confidence_pixels(pixel_count, max_pixels=5000):
    return min(1.0, pixel_count / max_pixels)  # Normaliza entre 0 y 1

def combined_confidence(bbox, pixel_count, img_shape, max_pixels=5000):
    score_area = compute_confidence_bbox(bbox, img_shape)
    score_pixels = compute_confidence_pixels(pixel_count, max_pixels)
    return (0.6 * score_area) + (0.4 * score_pixels)  # Ajusta pesos según necesidad

def binaryMaskIOU(mask1, mask2):
    mask1_area = np.count_nonzero(mask1 == 255)
    mask2_area = np.count_nonzero(mask2 == 255)
    intersection = np.count_nonzero(np.logical_and(mask1 == 255, mask2 == 255))
    union = mask1_area+mask2_area-intersection
    if union == 0: # Evitar dividir entre 0
        return 0
    iou = intersection/(union)
    return iou

def compute_ap_permuted(gt_boxes, pred_boxes):
    # With 3 or less bboxes, we do all possible permutations
    if len(pred_boxes) <= 3:
        all_aps = []
        for permuted_pred_boxes in itertools.permutations(pred_boxes):
            ap = compute_ap(gt_boxes, list(permuted_pred_boxes))
            all_aps.append(ap)
    else:
        # For more than 3 bboxes we do till 10 random permutations
        N = 10
        all_aps = []
        for _ in range(N):
            np.random.shuffle(pred_boxes) # Generate random order
            ap = compute_ap(gt_boxes, pred_boxes)  # Calcu AP with actual order
            all_aps.append(ap)

    return np.mean(all_aps)  # Mean over all permutations

def compute_ap(gt_boxes, pred_boxes):
    '''
    Extracted from Team6-2024 (https://github.com/mcv-m6-video/mcv-c6-2024-team6/blob/main/W1/task1/utils.py).
    '''
    # Initialize variables
    tp = np.zeros(len(pred_boxes))
    fp = np.zeros(len(pred_boxes))
    gt_matched = np.zeros(len(gt_boxes))

    if len(gt_boxes) == 0:
        ap = 0
    else:
        # Iterate over the predicted boxes
        for i, pred_box in enumerate(pred_boxes):
            ious = [binaryMaskIOU(pred_box, gt_box) for gt_box in gt_boxes]
            if len(ious) == 0:
                fp[i] = 1
                continue
            max_iou = max(ious)
            max_iou_idx = ious.index(max_iou)
            # print(f"IoU: {max_iou}")
            if max_iou >= 0.5 and not gt_matched[max_iou_idx]:
                tp[i] = 1
                gt_matched[max_iou_idx] = 1
            else:
                fp[i] = 1

        tp = np.cumsum(tp)
        fp = np.cumsum(fp)
        recall = tp / len(gt_boxes)
        precision = tp / (tp + fp)
        # Generate graph with the 11-point interpolated precision-recall curve
        recall_interp = np.linspace(0, 1, 11)
        precision_interp = np.zeros(11)
        for i, r in enumerate(recall_interp):
            array_precision = precision[recall >= r]
            if len(array_precision) == 0:
                precision_interp[i] = 0
            else:
                precision_interp[i] = max(precision[recall >= r])

        ap = np.mean(precision_interp)

    return ap

def create_mask_from_bbox(width, height, bbox, frame_id, obj_idx, dataset=None):
    """
    Create binary mask from bbox coords.
    """
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    if dataset=='gt':
        output_dir_bboxes_masks = os.path.join(output_dir, "bboxes_masks_gt")
    else:
        output_dir_bboxes_masks = os.path.join(output_dir, "bboxes_masks")
    os.makedirs(output_dir_bboxes_masks, exist_ok=True)
    
    mask = np.zeros((height, width), dtype=np.uint8)
    x, y, w, h = bbox

    mask[int(y):int(y + h), int(x):int(x + w)] = 255
    
    # cv2.imwrite(os.path.join(output_dir_bboxes_masks, f"bboxes_masks_{frame_id}_{obj_idx}.jpg"), mask)

    return mask

def load_boxes_from_txt(file_path):
    """
    Read bounding boxes from .txt and returns a dict
    where the key is the frame_id and the value is a list of bboxes as tuples (x, y, w, h).
    """
    boxes_per_frame = {}
    
    with open(file_path, 'r') as file:
        for line in file:
            data = line.strip().split(',')
            frame_id = int(data[0])
            x, y, w, h = int(data[2]), int(data[3]), int(data[4]), int(data[5])
            if frame_id not in boxes_per_frame:
                boxes_per_frame[frame_id] = []
            boxes_per_frame[frame_id].append((x, y, w, h))
            
    return boxes_per_frame

def calculate_mAP(aps):
    return np.mean(aps)


def convert_bbox_list_to_dict(bbox_info):
    """
    Conviert list of lists (output from read_ground_truth) in a dictionary
    where the key is the frame_id and the value is a list of bboxes as tuples (x, y, w, h)
    """
    bbox_dict = {}
    
    for frame_id, boxes in enumerate(bbox_info):
        bbox_dict[frame_id] = []
        for bbox in boxes:
            xtl, ytl, xbr, ybr = bbox
            x = xtl
            y = ytl
            w = xbr - xtl
            h = ybr - ytl
            bbox_dict[frame_id].append((x, y, w, h))
    
    return bbox_dict