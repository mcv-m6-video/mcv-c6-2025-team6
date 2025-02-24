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

def load_mean_variance(save_path):
    if os.path.exists(save_path):
        with open(save_path, "rb") as f:
            mean, variance = pickle.load(f)
        print("Cargando mean y variance desde el archivo guardado.")
        return mean, variance
    else:
        return None, None
    
def compute_mean_variance(image_folder):
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg'))]
    num_images = len(image_files)
    sample_size = int(25 * num_images / 100)
    sample_images = image_files[:sample_size]
    pixel_values = []
    
    for img_file in sample_images:
        img_path = os.path.join(image_folder, img_file)
        img = cv2.imread(img_path)
        
        if img is not None:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            pixel_values.append(img_gray.flatten())
    
    pixel_values = np.array(pixel_values)
    
    mean = np.mean(pixel_values, axis=0).reshape(img_gray.shape)
    variance = np.var(pixel_values, axis=0).reshape(img_gray.shape)
    print(f"Mean: {mean}")
    print(f"Variance: {variance}") 

    with open('mean_variance.pkl', "wb") as f:
        pickle.dump((mean, variance), f)
    
    return mean, variance

def update_background_model(img_gray, mean, variance, background_mask, rho=0.05):
    """
    Actualiza la media y varianza de fondo solo en los píxeles clasificados como fondo.
    """
    mean[background_mask] = rho * img_gray[background_mask] + (1 - rho) * mean[background_mask]
    variance[background_mask] = rho * (img_gray[background_mask] - mean[background_mask]) ** 2 + (1 - rho) * variance[background_mask]
    return mean, variance

def segment_foreground(image_folder, mean, variance, alpha=2.5, eps=50, min_samples=3, adaptive_segmentation=False):
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
    output_dir_clusters = os.path.join(output_dir, "bboxes_clusters")
    os.makedirs(output_dir_clusters, exist_ok=True)
    output_dir_masks = os.path.join(output_dir, "masks")
    os.makedirs(output_dir_masks, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_video = os.path.join(output_dir_videos, f"video_bbox-alph_{alpha}-eps_{eps}-ms_{min_samples}-adaptive_{adaptive_segmentation}.avi")
    video_writer = cv2.VideoWriter(output_video, fourcc, 20.0, (width, height))

    previous_bboxes = []
    bbox_id_counter = 1

    output_txt = open(os.path.join(output_dir, "bbox_results.txt"), "w")

    for img_file in tqdm(sample_images, desc="Processing Images", unit="image"):
        img_path = os.path.join(image_folder, img_file)
        img = cv2.imread(img_path)

        if img is None:
            continue

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1. Shadow detection
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(img_hsv)
        bd = (v / (s + 1e-6)).astype(np.float32)
        cd = np.abs(img_gray - mean)
        # Filter shadows and highlightings
        shadow_mask = (cd >= 10) & (bd > 5) & (bd < 12)
        # Combined mask
        exclusion_mask = shadow_mask

        # 2. Foreground segmentation
        foreground_mask = np.abs(img_gray - mean) >= alpha * (np.sqrt(variance) + 2)
        if adaptive_segmentation:
            background_mask = ~foreground_mask
            mean, variance = update_background_model(img_gray, mean, variance, background_mask)

        # Apply shadow mask
        foreground_mask[exclusion_mask] = 0

        # 3. Connected Components + Filter by Size
        num_labels, labeled_image, stats, _ = cv2.connectedComponentsWithStats(foreground_mask.astype(np.uint8), connectivity=8)
        filtered_labels = [i for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] > 50]
        filtered_mask = np.isin(labeled_image, filtered_labels).astype(np.uint8)

        # 4. Dilation
        kernel = np.ones((5, 5), np.uint8)
        redilated_mask = cv2.dilate(filtered_mask, kernel, iterations=2)

        # 5. Connected Components + Filter by Size
        num_labels_redilated, labeled_image_redilated, stats_redilated, _ = cv2.connectedComponentsWithStats(foreground_mask.astype(np.uint8), connectivity=8)
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
                    if pixel_count > 2000:
                        if (h/w < 1.1):
                            # print(f"Area ({x_min},{y_min}, {w}, {h}) of image {img_file}: {pixel_count}")
                            bbox_list.append([x, y, x + w, y + h])

        # 7. Color components
        colored_components = np.zeros_like(img)
        color = (155, 100, 0)
        for label_id in filtered_labels_redilated:
            colored_components[labeled_image_redilated == label_id] = color

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
                        margin = 15
                        if dist_x <= margin and dist_y <= margin:
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
                output_txt.write(f"{(img_file.split('.')[0]).split('_')[1]},{assigned_id},{int(x1_min)},{int(y1_min)},{int(x2_max)-int(x1_min)},{int(y2_max)-int(y1_min)},-1,-1,-1,-1\n")

            previous_bboxes = current_bboxes

            for bbox in current_bboxes:
                x1, y1, x2, y2, bbox_id = bbox
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(img, f'ID: {bbox_id}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                
                cv2.rectangle(final_mask, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(final_mask, f'ID: {bbox_id}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        cv2.imwrite(os.path.join(output_dir_clusters, f"clusters_detected_{img_file}"), img)
        cv2.imwrite(os.path.join(output_dir_masks, f"clusters_detected_{img_file}"), final_mask)

        # Blend original image with colored components
        blended = cv2.addWeighted(img, 0.7, colored_components, 0.5, 0)
        video_writer.write(blended)

    video_writer.release()
    output_txt.close()
    print("Processing completed.")
    print(f"Video saved in {output_video} and bbox results saved in bbox_results.txt")

if __name__ == "__main__":
    # extract_frames("AICity_data/train/S03/c010/vdo.avi", "frames_output", fps=10)

    mean, variance = pickle.load(open("mean_variance.pkl", "rb"))
    segment_foreground("frames_output", mean, variance, alpha=3.5, eps=150, min_samples=1, adaptive_segmentation=True)
    
    # Load GT bboxes
    xml_file = 'AICity_data/ai_challenge_s03_c010-full_annotation.xml'
    classes = ['car'] # The other class is bike
    path = 'frames_output/'
    frames_list = sorted(os.listdir(path))
    n_frames = len(frames_list)
    gt_boxes = read_ground_truth(xml_file, classes, n_frames)
    gt_boxes_per_frame = convert_bbox_list_to_dict(gt_boxes)
    # Load pred bboxes
    pred_boxes_per_frame = load_boxes_from_txt("output/bbox_results.txt")


    # Create masks for bounding boxes of GT and Pred
    filename_gt = "output/gt_masks_per_frame.pkl"
    if os.path.exists(filename_gt):
        with open(filename_gt, "rb") as f:
            gt_masks_per_frame = pickle.load(f)
    else:
        image_width, image_height = 1920, 1080
        gt_masks_per_frame = {frame_id: [create_mask_from_bbox(image_width, image_height, bbox, frame_id, idx, dataset='gt') for idx, bbox in enumerate(boxes)] 
                        for frame_id, boxes in gt_boxes_per_frame.items()}
        with open(filename_gt, "wb") as f:
            pickle.dump(gt_masks_per_frame, f)

    
    filename_pred = "output/pred_masks_per_frame.pkl"
    if os.path.exists(filename_pred):
        with open(filename_pred, "rb") as f:
            pred_masks_per_frame = pickle.load(f)
    else:
        image_width, image_height = 1920, 1080
        pred_masks_per_frame = {frame_id: [create_mask_from_bbox(image_width, image_height, bbox, frame_id, idx) for idx, bbox in enumerate(boxes)] 
                                for frame_id, boxes in pred_boxes_per_frame.items()}
        with open(filename_pred, "wb") as f:
            pickle.dump(pred_masks_per_frame, f)

    aps = []
    start_frame = int(len(gt_masks_per_frame)*0.25)
    # Calc AP for each frame comparing predictions with GT
    for frame_id in gt_boxes_per_frame.keys():
        if frame_id < start_frame:
            continue
        
        gt_masks = gt_masks_per_frame[frame_id]

        if frame_id in pred_boxes_per_frame:
            pred_masks = pred_masks_per_frame[frame_id]    
        else:
            pred_masks = []
        
        if len(gt_masks) > 0 or len(pred_masks) > 0:
            ap = compute_ap(gt_masks, pred_masks)
            aps.append(ap)
            # print(f"AP para el frame {frame_id}: {ap}")
    
    # Calcular el mAP usando la función calculate_mAP
    mAP = calculate_mAP(aps)
    print(f"mAP promedio: {mAP}")

