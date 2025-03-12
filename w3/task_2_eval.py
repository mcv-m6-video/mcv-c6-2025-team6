import xml.etree.ElementTree as ET
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment


# Parse the XML file to extract ground truth data
def parse_xml_annotations(file_path):
    """
    Parse the ground truth XML file.
    Returns a dictionary with {frame: [(id, x, y, w, h)]}
    """
    tree = ET.parse(file_path)
    root = tree.getroot()

    ground_truths = {}

    for track in root.findall("track"):
        track_id = track.attrib['id']
        for box in track.findall("box"):
            frame = int(box.attrib['frame'])
            xtl, ytl, xbr, ybr = map(float, [box.get("xtl"), box.get("ytl"), box.get("xbr"), box.get("ybr")])
            w, h = int(xbr - xtl), int(ybr - ytl)
            x, y = int(xtl), int(ytl)

            if frame not in ground_truths:
                ground_truths[frame] = []
            ground_truths[frame].append((track_id, x, y, w, h))

    return ground_truths


# Parse the MOTS predictions file
def parse_mots_txt(file_path):
    """
    Parse the MOTS prediction file (txt format).
    Returns a dictionary with {frame: [(id, x, y, w, h)]}
    """
    predictions = {}
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            #print(parts)
            if len(parts) < 6:
                continue  # Skip invalid lines

            frame, track_id, x, y, w, h = map(int, parts[:6])

            if frame not in predictions:
                predictions[frame] = []
            predictions[frame].append((track_id, x, y, w, h))

    return predictions


# Calculate ID metrics
def calculate_id_metrics(gt_boxes, pred_boxes):
    """
    Calculate Identity tracking metrics: IDTP, IDFP, IDFN.
    """
    if not gt_boxes or not pred_boxes:
        return 0, len(pred_boxes), len(gt_boxes)

    cost_matrix = np.zeros((len(gt_boxes), len(pred_boxes)))

    for i, gt in enumerate(gt_boxes):
        for j, pred in enumerate(pred_boxes):
            # Compute Euclidean distance
            gt_center = np.array([(gt[1] + gt[3] / 2), (gt[2] + gt[4] / 2)])
            pred_center = np.array([(pred[1] + pred[3] / 2), (pred[2] + pred[4] / 2)])
            cost_matrix[i, j] = np.linalg.norm(gt_center - pred_center)

    # Solve assignment problem using Hungarian algorithm
    gt_indices, pred_indices = linear_sum_assignment(cost_matrix)

    matched_pairs = [(gt_boxes[i], pred_boxes[j]) for i, j in zip(gt_indices, pred_indices)]
    idtp = len(matched_pairs)
    idfp = len(pred_boxes) - idtp
    idfn = len(gt_boxes) - idtp

    return idtp, idfp, idfn


# Compute IDF1 score
def calculate_idf1(idtp, idfp, idfn):
    """
    Compute IDF1 score.
    """
    try:
        return (2 * idtp) / (2 * idtp + idfp + idfn)
    except ZeroDivisionError:
        return 0.0


# Compute HOTA score
def calculate_hota(idtp, idfp, idfn):
    """
    Compute HOTA score.
    """
    try:
        deta=idtp/(idtp + idfp + idfn)
        assa=idtp/(idtp + 0.5 * (idfp + idfn))
        return np.sqrt(deta*assa)
    except ZeroDivisionError:
        return 0.0


# Main function to evaluate tracking performance
def evaluate_tracking(gt_file, pred_file):
    """
    Evaluate tracking results using HOTA and ID metrics.
    """
    gt_data = parse_xml_annotations(gt_file)
    # for clave, valor in gt_data.items():
    #     print(f'{clave}: {valor}')
    pred_data = parse_mots_txt(pred_file)

    idf1_scores = []
    hota_scores = []

    all_frames = sorted(set(gt_data.keys()).union(pred_data.keys()))

    for frame in all_frames:
        gt_boxes = gt_data.get(frame, [])
        pred_boxes = pred_data.get(frame, [])

        idtp, idfp, idfn = calculate_id_metrics(gt_boxes, pred_boxes)

        idf1 = calculate_idf1(idtp, idfp, idfn)
        hota = calculate_hota(idtp, idfp, idfn)

        idf1_scores.append(idf1)
        hota_scores.append(hota)

    avg_idf1 = np.mean(idf1_scores)
    avg_hota = np.mean(hota_scores)

    print(f"Average IDF1 Score: {avg_idf1:.4f}")
    print(f"Average HOTA Score: {avg_hota:.4f}")


if __name__ == "__main__":
    gt_file = "../AICity_data/ai_challenge_s03_c010-full_annotation.xml"
    pred_file = "tracked_objects_log.txt"

    evaluate_tracking(gt_file, pred_file)