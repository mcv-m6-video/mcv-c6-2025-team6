import cv2
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
from utils import *
import numpy as np
import pickle
import random
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from collections import Counter
import itertools

def load_mean_variance(save_path):
    if os.path.exists(save_path):
        with open(save_path, "rb") as f:
            mean, variance, mean_s, mean_v = pickle.load(f)
        print("Cargando mean y variance desde el archivo guardado.")
        return mean, variance, mean_s, mean_v
    else:
        return None, None
    
def compute_mean_variance(image_folder):
    """
    Calculates the mean and variance of pixel values in grayscale,
    as well as the mean of the S and V channels in the HSV color space.

    Args:

    image_folder (str): Path to the folder containing images.
    Returns:

    tuple: (mean, variance, mean_s, mean_v).
    """
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
    num_images = len(image_files)
    sample_size = int(25 * num_images / 100)
    sample_images = image_files[:sample_size]

    pixel_values_gray = []
    pixel_values_s = []
    pixel_values_v = []

    for img_file in sample_images:
        img_path = os.path.join(image_folder, img_file)
        img = cv2.imread(img_path)

        if img is not None:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            _, s, v = cv2.split(img_hsv)

            pixel_values_gray.append(img_gray.flatten())
            pixel_values_s.append(s.flatten())
            pixel_values_v.append(v.flatten())

    pixel_values_gray = np.array(pixel_values_gray)
    pixel_values_s = np.array(pixel_values_s)
    pixel_values_v = np.array(pixel_values_v)

    mean = np.mean(pixel_values_gray, axis=0).reshape(img_gray.shape)
    variance = np.var(pixel_values_gray, axis=0).reshape(img_gray.shape)
    mean_s = np.mean(pixel_values_s, axis=0).reshape(img_gray.shape)
    mean_v = np.mean(pixel_values_v, axis=0).reshape(img_gray.shape)

    print(f"Mean (Grayscale): {mean}")
    print(f"Variance (Grayscale): {variance}")
    print(f"Mean S (HSV): {mean_s}")
    print(f"Mean V (HSV): {mean_v}")

    with open('output/mean_variance.pkl', "wb") as f:
        pickle.dump((mean, variance, mean_s, mean_v), f)

    return mean, variance, mean_s, mean_v

def update_background_model(img_gray, img_s, img_v, mean, variance, mean_s, mean_v, background_mask, rho=0.05):
    """
    Updates the background mean and variance for the grayscale channel (Gray)
    and the S and V channels in the HSV color space, only for pixels classified as background.

    Args:
        img_gray (np.array): Grayscale image.
        img_s (np.array): S channel of the HSV image.
        img_v (np.array): V channel of the HSV image.
        mean (np.array): Background mean in grayscale.
        variance (np.array): Background variance in grayscale.
        mean_s (np.array): Background mean in the S channel.
        mean_v (np.array): Background mean in the V channel.
        background_mask (np.array): Boolean mask where True indicates background.
        rho (float): Learning rate.

    Returns:
        tuple: (mean, variance, mean_s, mean_v) updated.
    """
    mean[background_mask] = rho * img_gray[background_mask] + (1 - rho) * mean[background_mask]
    variance[background_mask] = rho * (img_gray[background_mask] - mean[background_mask]) ** 2 + (1 - rho) * variance[background_mask]

    mean_s[background_mask] = rho * img_s[background_mask] + (1 - rho) * mean_s[background_mask]
    mean_v[background_mask] = rho * img_v[background_mask] + (1 - rho) * mean_v[background_mask]

    return mean, variance, mean_s, mean_v

def segment_foreground(image_folder, mean, variance, mean_s, mean_v, gt_boxes_per_frame, alpha=2.5, eps=50, min_samples=3, adaptive_segmentation=False, margin_overlap=15, min_area=2000):
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
    num_images = len(image_files)
    sample_size = int(25 * num_images / 100)
    sample_images = image_files[sample_size:]

    first_img_path = os.path.join(image_folder, sample_images[0])
    first_img = cv2.imread(first_img_path)
    height, width, _ = first_img.shape

    # Output dirs
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    output_dir_videos = os.path.join(output_dir, "videos")
    os.makedirs(output_dir_videos, exist_ok=True)
    output_dir_clusters = os.path.join(output_dir, f"bboxes_clusters_-alph_{alpha}-eps_{eps}-mov_{margin_overlap}-ma_{min_area}-adaptive_{adaptive_segmentation}")
    os.makedirs(output_dir_clusters, exist_ok=True)
    output_dir_masks = os.path.join(output_dir, f"masks_-alph_{alpha}-eps_{eps}-mov_{margin_overlap}-ma_{min_area}-adaptive_{adaptive_segmentation}")
    os.makedirs(output_dir_masks, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_video = os.path.join(output_dir_videos, f"video_bbox-alph_{alpha}-eps_{eps}-mov_{margin_overlap}-ma_{min_area}-adaptive_{adaptive_segmentation}.avi")
    video_writer = cv2.VideoWriter(output_video, fourcc, 20.0, (width, height))

    previous_bboxes = []
    bbox_id_counter = 1
    bbox_results = f"bbox_results-alph_{alpha}-eps_{eps}-mov_{margin_overlap}-ma_{min_area}-adaptive_{adaptive_segmentation}.txt"
    output_txt = open(os.path.join(output_dir, bbox_results), "w")

    for img_file in tqdm(sample_images, desc="Processing Images", unit="image"):
        img_path = os.path.join(image_folder, img_file)
        img = cv2.imread(img_path)

        if img is None:
            continue

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1. Shadow detection
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(img_hsv)
        SD = np.abs(s - mean_s) / (mean_s + 1e-6)
        BD = v / (mean_v + 1e-6)
        shadow_mask = (SD <= 3) & (0.4 <= BD) & (BD <= 0.9)

        # 2. Foreground segmentation
        foreground_mask = np.abs(img_gray - mean) >= alpha * (np.sqrt(variance) + 2)
        if adaptive_segmentation:
            background_mask = ~foreground_mask
            mean, variance, mean_s, mean_v = update_background_model(img_gray, s, v, mean, variance, mean_s, mean_v, background_mask)

        # Apply shadow mask
        foreground_mask[shadow_mask] = 0

        # 3. Connected Components + Filter by Size
        num_labels, labeled_image, stats, _ = cv2.connectedComponentsWithStats(foreground_mask.astype(np.uint8), connectivity=8)
        filtered_labels = [i for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] > 50]
        filtered_mask = np.isin(labeled_image, filtered_labels).astype(np.uint8)

        # 4. Dilation
        kernel = np.ones((5, 5), np.uint8)
        redilated_mask = cv2.dilate(filtered_mask, kernel, iterations=2)

        # 5. Connected Components + Filter by Size
        num_labels_redilated, labeled_image_redilated, stats_redilated, _ = cv2.connectedComponentsWithStats(redilated_mask.astype(np.uint8), connectivity=8)
        filtered_labels_redilated = [i for i in range(1, num_labels_redilated) if stats_redilated[i, cv2.CC_STAT_AREA] > 500]
        final_mask = np.isin(labeled_image_redilated, filtered_labels_redilated).astype(np.uint8)

        bbox_list = []
        # 6. Clustering for connected components (DBSCAN)
        centroids = np.array([
            [stats_redilated[i, cv2.CC_STAT_LEFT] + stats_redilated[i, cv2.CC_STAT_WIDTH] // 2,
            stats_redilated[i, cv2.CC_STAT_TOP] + stats_redilated[i, cv2.CC_STAT_HEIGHT] // 2]
            for i in filtered_labels_redilated
        ])
        if centroids.size > 0:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(centroids)
            labels = dbscan.labels_
            
            output_image = cv2.cvtColor(final_mask * 255, cv2.COLOR_GRAY2BGR)

            clusters = {}
            for i, label in enumerate(labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(filtered_labels_redilated[i])

            # Draw clusters' bounding boxes
            for cluster_labels in clusters.values():
                x_min = min(stats_redilated[i, cv2.CC_STAT_LEFT] for i in cluster_labels)
                y_min = min(stats_redilated[i, cv2.CC_STAT_TOP] for i in cluster_labels)
                x_max = max(stats_redilated[i, cv2.CC_STAT_LEFT] + stats_redilated[i, cv2.CC_STAT_WIDTH] for i in cluster_labels)
                y_max = max(stats_redilated[i, cv2.CC_STAT_TOP] + stats_redilated[i, cv2.CC_STAT_HEIGHT] for i in cluster_labels)

                # Extract bounding box region from binary mask and count mask pixels inside
                region = final_mask[y_min:y_max, x_min:x_max]
                pixel_count = np.sum(region == 1)
                
                x, y, w, h = x_min, y_min, (x_max-x_min), (y_max-y_min)
                if w > 60 or h > 60:
                    if pixel_count > min_area:
                        if (h/w < 1.1):
                            # print(f"Area ({x_min},{y_min}, {w}, {h}) of image {img_file}: {pixel_count}")
                            bbox_list.append([x, y, x + w, y + h])

        # 7. Color components
        colored_components = np.zeros_like(img)
        color = (155, 100, 0)
        for label_id in filtered_labels_redilated:
            colored_components[labeled_image_redilated == label_id] = color

        frame_id_str = (img_file.split('.')[0]).split('_')[1]
        # 8. Final filtering of bboxes
        if bbox_list:
            bbox_list = np.array(bbox_list)
            bbox_list = bbox_list[np.argsort((bbox_list[:, 2] - bbox_list[:, 0]) * (bbox_list[:, 3] - bbox_list[:, 1]))[::-1]]

            # Filter small bbox inside others or expand bbox if overlap
            filtered_bbox_list = []
            for bbox in bbox_list:
                x1, y1, x2, y2 = bbox
                is_inside = False
                for i, other_bbox in enumerate(filtered_bbox_list):
                    ox1, oy1, ox2, oy2, _ = other_bbox
                    if not (x2 < ox1 or x1 > ox2 or y2 < oy1 or y1 > oy2):
                        new_x1 = min(x1, ox1)
                        new_y1 = min(y1, oy1)
                        new_x2 = max(x2, ox2)
                        new_y2 = max(y2, oy2)

                        region = final_mask[new_y1:new_y2, new_x1:new_x2]
                        pixel_count = np.sum(region == 1)
                        confidence_score = combined_confidence([new_x1, new_y1, new_x2, new_y2], pixel_count, final_mask.shape)
            
                        filtered_bbox_list[i] = [new_x1, new_y1, new_x2, new_y2, confidence_score]
                        is_inside = True

                        break
                    else:
                        # Calc distance between bboxes and verify if it is lower than margin
                        dist_x = max(0, ox1 - x2) if ox1 > x2 else max(0, x1 - ox2)
                        dist_y = max(0, oy1 - y2) if oy1 > y2 else max(0, y1 - oy2)
                        if dist_x <= margin_overlap and dist_y <= margin_overlap:
                            # Expand bbox to cover both
                            new_x1 = min(x1, ox1)
                            new_y1 = min(y1, oy1)
                            new_x2 = max(x2, ox2)
                            new_y2 = max(y2, oy2)
                            
                            region = final_mask[new_y1:new_y2, new_x1:new_x2]
                            pixel_count = np.sum(region == 1)
                            confidence_score = combined_confidence([new_x1, new_y1, new_x2, new_y2], pixel_count, final_mask.shape)

                            filtered_bbox_list[i] = [new_x1, new_y1, new_x2, new_y2, confidence_score]
                            is_inside = True
                            break
                
                if not is_inside:
                    region = final_mask[y1:y2, x1:x2]
                    pixel_count = np.sum(region == 1)
                    confidence_score = combined_confidence([x1, y1, x2, y2], pixel_count, final_mask.shape)

                    filtered_bbox_list.append([x1, y1, x2, y2, confidence_score])

            bbox_list = np.array(filtered_bbox_list)
            indices_ordenados = np.argsort(bbox_list[:, -1])[::-1]
            bbox_list = bbox_list[indices_ordenados]    # Sort by confidence score
            
            final_mask = final_mask * 255

            current_bboxes = []
            for cluster_id in range(bbox_list.shape[0]):
                cluster_bboxes = bbox_list[cluster_id,:]
                x1_min, y1_min, x2_max, y2_max = np.min(cluster_bboxes[0]), np.min(cluster_bboxes[1]), np.max(cluster_bboxes[2]), np.max(cluster_bboxes[3])

                assigned_id = None
                for prev_bbox in previous_bboxes:
                    prev_cx, prev_cy = (prev_bbox[0] + prev_bbox[2]) / 2, (prev_bbox[1] + prev_bbox[3]) / 2
                    cx, cy = (x1_min + x2_max) / 2, (y1_min + y2_max) / 2
                    distance = np.sqrt((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2)

                    if distance < 80:
                        assigned_id = prev_bbox[4]
                        break

                if assigned_id is None:
                    assigned_id = bbox_id_counter
                    bbox_id_counter += 1

                current_bboxes.append([x1_min, y1_min, x2_max, y2_max, assigned_id])
                output_txt.write(f"{frame_id_str},{assigned_id},{int(x1_min)},{int(y1_min)},{int(x2_max)-int(x1_min)},{int(y2_max)-int(y1_min)},-1,-1,-1,-1\n")

            previous_bboxes = current_bboxes

            for bbox in current_bboxes:
                x1, y1, x2, y2, bbox_id = bbox
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(img, f'ID: {bbox_id}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                cv2.rectangle(final_mask, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(final_mask, f'ID: {bbox_id}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        # cv2.imwrite(os.path.join(output_dir_clusters, f"clusters_detected_{img_file}"), img)
        # cv2.imwrite(os.path.join(output_dir_masks, f"clusters_detected_{img_file}"), final_mask)
        
        for gt_bbox in gt_boxes_per_frame[int(frame_id_str)-1]:
            x1, y1, w, h = gt_bbox
            x2, y2 = x1 + w, y1 + h
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)  # Azul para GT
    
        # Blend original image with colored components
        blended = cv2.addWeighted(img, 0.7, colored_components, 0.5, 0)
        video_writer.write(blended)

    video_writer.release()
    output_txt.close()
    print("Processing completed.")
    print(f"Video saved in {output_video} and bbox results saved in {bbox_results}")

if __name__ == "__main__":
    # extract_frames("AICity_data/train/S03/c010/vdo.avi", "frames_output", fps=10)

    mean, variance, mean_s, mean_v = pickle.load(open("output/mean_variance.pkl", "rb"))

    alpha_values = [6]
    eps_values = [80]
    adaptive_options = [True]
    margins_overlap = [10]
    min_areas = [3500]

    results_file = "output/results.txt"
    os.makedirs("output", exist_ok=True)

    with open(results_file, "a") as f:
        for alpha, eps, adaptive_segmentation, margin_overlap, min_area in itertools.product(alpha_values, eps_values, adaptive_options, margins_overlap, min_areas):
            print(f"\n=== Ejecutando con alpha={alpha}, eps={eps}, adaptive_segmentation={adaptive_segmentation}, margin_overlap={margin_overlap}, min_area={min_area} ===\n")
            
            # Load GT bboxes
            xml_file = 'AICity_data/ai_challenge_s03_c010-full_annotation.xml'
            classes = ['car'] # The other class is bike
            path = 'frames_output/'
            frames_list = sorted(os.listdir(path))
            n_frames = len(frames_list)
            gt_boxes = read_ground_truth(xml_file, classes, n_frames)
            gt_boxes_per_frame = convert_bbox_list_to_dict(gt_boxes)

            segment_foreground("frames_output", mean, variance, mean_s, mean_v, gt_boxes_per_frame, alpha=alpha, eps=eps, min_samples=1, adaptive_segmentation=adaptive_segmentation, margin_overlap=margin_overlap, min_area=min_area)

            # Load pred bboxes
            bbox_results = f"bbox_results-alph_{alpha}-eps_{eps}-mov_{margin_overlap}-ma_{min_area}-adaptive_{adaptive_segmentation}.txt"
            pred_boxes_per_frame = load_boxes_from_txt(os.path.join("output", bbox_results))

            # Create masks for bounding boxes of GT and Pred
            image_width, image_height = 1920, 1080
            gt_masks_per_frame = {frame_id: [create_mask_from_bbox(image_width, image_height, bbox, frame_id, idx, dataset='gt') for idx, bbox in enumerate(boxes)] 
                            for frame_id, boxes in gt_boxes_per_frame.items()}
            
            image_width, image_height = 1920, 1080
            pred_masks_per_frame = {frame_id: [create_mask_from_bbox(image_width, image_height, bbox, frame_id, idx) for idx, bbox in enumerate(boxes)] 
                                    for frame_id, boxes in pred_boxes_per_frame.items()}

            aps = []
            start_frame = int(len(gt_masks_per_frame)*0.25)
            # Calc AP for each frame comparing predictions with GT
            for frame_id in tqdm(gt_boxes_per_frame.keys(), desc="Processing frames APs", position=0, leave=True):
                if frame_id < start_frame:
                    continue
                
                gt_masks = gt_masks_per_frame[frame_id]

                if frame_id in pred_boxes_per_frame:
                    pred_masks = pred_masks_per_frame[frame_id]
                else:
                    pred_masks = []
                
                if len(gt_masks) > 0 or len(pred_masks) > 0:
                    ap = compute_ap_permuted(gt_masks, pred_masks)
                    aps.append(ap)
                    # print(f"AP para el frame {frame_id}: {ap}")
            
            # Calculate mAP
            mAP = calculate_mAP(aps)
            print(f"mAP = {mAP} (alpha={alpha}, eps={eps}, adaptive_segmentation={adaptive_segmentation}, margin_overlap={margin_overlap}, min_area={min_area})\n")
            f.write(f"mAP = {mAP} (alpha={alpha}, eps={eps}, adaptive_segmentation={adaptive_segmentation}, margin_overlap={margin_overlap}, min_area={min_area})\n")

