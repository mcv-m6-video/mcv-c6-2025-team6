import cv2
import numpy as np
import json
import os
from sort import Sort
from utils import *

# This script processes video frames, tracks objects using the SORT tracker, 
# optionally filters out overlapping bounding boxes, and saves the results as images and/or video. 
# The script also calculates the movement of tracked objects and assigns unique IDs and colors.

def process_frames_and_save_results(input_folder, txt_file, output_folder, overlap_threshold=0.9, output_video=False, movement_threshold=3.0, static_frame_threshold=2000):
    frame_data = read_bounding_boxes_from_file(txt_file)
    frame_numbers = sorted(frame_data.keys())
    
    all_frame_data = []
    
    if output_video:
        first_frame = cv2.imread(os.path.join(input_folder, f"frame_{frame_numbers[0]:04d}.jpg"))
        frame_height, frame_width = first_frame.shape[:2]
        video_writer = cv2.VideoWriter(f"{output_folder}/tracked_video.avi", cv2.VideoWriter_fourcc(*'XVID'), 30, (frame_width, frame_height))
    
    tracker = Sort(max_age=10, min_hits=3)
    
    track_colors = {}
    
    previous_positions = {}

    txt_output_file = os.path.join(output_folder, "tracking_data.txt")
    with open(txt_output_file, 'w') as f:
        for frame_num in frame_numbers:
            frame = cv2.imread(os.path.join(input_folder, f"frame_{frame_num:04d}.jpg"))
            if frame is None:
                continue
            
            current_boxes = frame_data[frame_num]
            
            filtered_boxes = filter_overlapping_boxes_based_on_iou(current_boxes, iou_threshold=overlap_threshold)
            
            detections = np.array([[x, y, x + w, y + h] for x, y, w, h in filtered_boxes])
            
            trackers = tracker.update(detections)
            
            frame_with_boxes = frame.copy()
            
            for d in trackers:
                d = d.astype(np.int32)
                x1, y1, x2, y2, track_id = d
                
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                
                if track_id not in previous_positions:
                    previous_positions[track_id] = []
                
                previous_positions[track_id].append(center)
                
                if len(previous_positions[track_id]) > static_frame_threshold:
                    previous_positions[track_id].pop(0)
                
                if len(previous_positions[track_id]) > 1:
                    movement = calculate_euclidean_distance_between_points(previous_positions[track_id][-2], previous_positions[track_id][-1])
                    if movement < movement_threshold:
                        continue

                if track_id not in track_colors:
                    track_colors[track_id] = generate_random_color_for_object()
                
                color = track_colors[track_id]
                
                cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), color, 2)
                
                text_size = cv2.getTextSize(f"ID: {track_id}", cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
                text_x1 = x1
                text_y1 = y1 - 30
                text_x2 = x1 + text_size[0] + 10
                text_y2 = y1 - 10
                
                cv2.rectangle(frame_with_boxes, (text_x1, text_y1), (text_x2, text_y2), color, -1)
                cv2.putText(frame_with_boxes, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            
                # tracking data (frame_id, track_id, x, y, w, h, confidence) to txt file
                width = x2 - x1
                height = y2 - y1
                confidence = 1.0  # the SORT tracker doesn't give a confidence value, we'll use a fixed value (1.0)

                f.write(f"{frame_num}, {track_id}, {x1}, {y1}, {width}, {height}, {confidence}\n")
            
            frame_filename = os.path.join(output_folder, f"frame_{frame_num}.jpg")
            cv2.imwrite(frame_filename, frame_with_boxes)
            
            all_frame_data.append({
                "frame": frame_num,
                "boxes": [{"id": int(d[4]), "box": [int(d[0]), int(d[1]), int(d[2] - d[0]), int(d[3] - d[1])]} for d in trackers]
            })
            
            if output_video:
                video_writer.write(frame_with_boxes)
        
        if output_video:
            video_writer.release()
        
        with open(os.path.join(output_folder, "tracked_data.json"), 'w') as json_file:
            json.dump(all_frame_data, json_file, indent=4)
    
    print(f"Processed frames saved to {output_folder} and tracking data saved to {txt_output_file}")


if __name__ == "__main__":
    input_folder = "C:/Users/laila/mcv-c6-2025-team6/output_frames"  
    txt_file = "C:/Users/laila/c6-/pred_bbox_2.txt"  
    output_folder = "output_frames2.2"  

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    process_frames_and_save_results(input_folder, txt_file, output_folder, output_video=True)
    print("processing complete..")
