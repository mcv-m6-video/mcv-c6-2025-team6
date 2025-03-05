import cv2
import numpy as np
import json
import os
from utils import *  

# This script processes video frames, tracks objects using the Maximum Overlap Technique, 
# optionally filters out overlapping bounding boxes, and saves the results as images and/or video. 
# The script also calculates the movement of tracked objects and assigns unique IDs and colors.

def filter_non_moving_objects(previous_positions, current_position, track_id, movement_threshold=3.0, static_frame_threshold=2000):

    if track_id not in previous_positions:
        previous_positions[track_id] = []
    
    previous_positions[track_id].append(current_position)

    if len(previous_positions[track_id]) > static_frame_threshold:
        previous_positions[track_id].pop(0)

    if len(previous_positions[track_id]) > 1:
        movement = calculate_euclidean_distance_between_points(previous_positions[track_id][-2], previous_positions[track_id][-1])
        if movement < movement_threshold:
            return False  # Object non-moving
    return True  # Object moving


def track_objects(previous_boxes, current_boxes, prev_ids, threshold_iou=0.4, overlap_threshold=0.9):
    new_ids = []
    assigned = []
    next_available_id = max(prev_ids) + 1 if prev_ids else 1

    for current_box in current_boxes:
        matched = False
        for i, prev_box in enumerate(previous_boxes):
            iou = calculate_iou_between_boxes(prev_box, current_box)
            if iou >= threshold_iou:
                # we found a match, assign the previous ID to the current box
                new_ids.append(prev_ids[i])
                assigned.append(i)
                matched = True
                break
        
        if not matched:
            new_ids.append(next_available_id)
            prev_ids.append(next_available_id)
            next_available_id += 1

    final_ids = []
    final_boxes = []
    for i, current_box in enumerate(current_boxes):
        overlap_found = False
        for j, final_box in enumerate(final_boxes):
            if calculate_iou_between_boxes(final_box, current_box) >= overlap_threshold:
                overlap_found = True
                break
        if not overlap_found:
            final_boxes.append(current_box)
            final_ids.append(new_ids[i])

    return final_ids, final_boxes


def process_frames_and_save(input_folder, txt_file, output_folder, threshold_iou=0.4, overlap_threshold=0.9, movement_threshold=3.0, static_frame_threshold=2000, output_video=False):
    frame_data = read_bounding_boxes_from_file(txt_file)
    frame_numbers = sorted(frame_data.keys())

    all_frame_data = []
    
    if output_video:
        first_frame = cv2.imread(os.path.join(input_folder, f"frame_{frame_numbers[0]:04d}.jpg"))
        frame_height, frame_width = first_frame.shape[:2]
        video_writer = cv2.VideoWriter(f"{output_folder}/tracked_video.avi", cv2.VideoWriter_fourcc(*'XVID'), 30, (frame_width, frame_height))

    prev_ids = []
    prev_boxes = []
    previous_positions = {}  
    object_colors = {}  

    output_txt_file = os.path.join(output_folder, "tracking_data.txt")
    with open(output_txt_file, "w") as f:
        for frame_num in frame_numbers:
            frame = cv2.imread(os.path.join(input_folder, f"frame_{frame_num:04d}.jpg"))
            current_boxes = frame_data[frame_num]

            filtered_boxes = filter_overlapping_boxes_based_on_iou(current_boxes, iou_threshold=overlap_threshold)
            
            if frame_num == frame_numbers[0]:
                tracked_ids = list(range(1, len(filtered_boxes) + 1))
            else:
                tracked_ids, filtered_boxes = track_objects(prev_boxes, filtered_boxes, prev_ids, threshold_iou, overlap_threshold)

            frame_with_boxes = frame.copy()

            for i, box in enumerate(filtered_boxes):

                x, y, w, h = box[:4]
                center = (x + w // 2, y + h // 2)

                if not filter_non_moving_objects(previous_positions, center, tracked_ids[i], movement_threshold, static_frame_threshold):
                    continue  # if object has not moved , skip

                confidence = box[-1] if len(box) > 4 else 1.0  

                object_id = tracked_ids[i]
                if object_id not in object_colors:
                    object_colors[object_id] = generate_random_color_for_object()
                color = object_colors[object_id]
                
                cv2.rectangle(frame_with_boxes, (x, y), (x + w, y + h), color, 2)

                label = f"ID: {object_id}"
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
                text_w, text_h = text_size
                cv2.rectangle(frame_with_boxes, (x, y - text_h - 5), (x + text_w, y), color, -1)
                cv2.putText(frame_with_boxes, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                
                f.write(f"{frame_num}, {object_id}, {x}, {y}, {w}, {h}, {confidence:.4f}\n")

            frame_filename = os.path.join(output_folder, f"frame_{frame_num:04d}.jpg")
            cv2.imwrite(frame_filename, frame_with_boxes)
            all_frame_data.append({"frame": frame_num, "boxes": [{"id": tracked_ids[i], "box": filtered_boxes[i]} for i in range(len(filtered_boxes))]})

            if output_video:
                video_writer.write(frame_with_boxes)

            prev_boxes = filtered_boxes
            prev_ids = tracked_ids

        if output_video:
            video_writer.release()

    with open(os.path.join(output_folder, "tracked_data.json"), 'w') as json_file:
        json.dump(all_frame_data, json_file, indent=4)

    print(f"Processed frames saved to {output_folder}")
    print(f"Tracking data saved to {output_txt_file}")


if __name__ == "__main__":
    input_folder = "C:/Users/laila/mcv-c6-2025-team6/output_frames"  
    txt_file = "C:/Users/laila/c6-/pred_bbox_2.txt"  
    output_folder = "output_frames2.1"  

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    process_frames_and_save(input_folder, txt_file, output_folder, output_video=True)
    print("Processing complete..")
