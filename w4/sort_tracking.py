import cv2
import numpy as np
import os
from task_2_sort import Sort
from task_2_utils import *

from utils import get_next_experiment_folder
from metrics import get_hota_idf1

# This script processes video frames, tracks objects using the SORT tracker,
# optionally filters out overlapping bounding boxes, and saves the results as images and/or video.
# The script also calculates the movement of tracked objects and assigns unique IDs and colors.

def process_video_and_save_results(input_video, txt_file, output_folder, overlap_threshold=0.9, output_video=False, save_frames=False, movement_threshold=3.0, static_frame_threshold=2000):

    frame_data = read_bounding_boxes_from_file(txt_file)
    frame_numbers = sorted(frame_data.keys())
    
    video_capture = cv2.VideoCapture(input_video)
    if not video_capture.isOpened():
        print("Error: Could not open video file.")
        return
    
    output_folder = get_next_experiment_folder(output_folder, name="tracking")
    os.makedirs(output_folder, exist_ok=True)

    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = video_capture.get(cv2.CAP_PROP_FPS)

    if output_video:
        video_writer = cv2.VideoWriter(f"{output_folder}/tracked_video.avi", cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))
    
    tracker = Sort(max_age=30, min_hits=2)
    
    track_colors = {}
    previous_positions = {}

    txt_output_file = os.path.join(output_folder, "tracking_data.txt")
    with open(txt_output_file, 'w') as f:
        frame_num = 0  
        
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            if frame_num in frame_data:
                current_boxes = frame_data[frame_num]
                filtered_boxes = filter_overlapping_boxes_based_on_iou(current_boxes, iou_threshold=overlap_threshold)
                
                detections = np.array([[x1, y1, x1+w, y1+h] for x1, y1, w, h in filtered_boxes])
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
                
                    # (frame_id, track_id, x, y, w, h)
                    width = x2 - x1
                    height = y2 - y1

                    f.write(f"{frame_num}, {track_id}, {x1}, {y1}, {width}, {height}, 1,-1,-1,-1\n")
                
                if save_frames:
                    frame_filename = os.path.join(f"{output_folder}/frames", f"frame_{frame_num:04d}.jpg")
                    cv2.imwrite(frame_filename, frame_with_boxes)
                
                if output_video:
                    video_writer.write(frame_with_boxes)
            
            frame_num += 1
        
        if output_video:
            video_writer.release()

    video_capture.release()
    print(f"Processed video saved to {output_folder} and tracking data saved to {txt_output_file}")


if __name__ == "__main__":
    input_video = "/home/maria/mcv-c6-2025-team6/data/aic19-track1-mtmc-train/train/S01/c001/vdo.avi"  
    txt_file = "/home/maria/mcv-c6-2025-team6/w4/output/runs/_predict1/vdo.txt"  
    output_folder = "/home/maria/mcv-c6-2025-team6/w4/output/sort"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    process_video_and_save_results(input_video, txt_file, output_folder, output_video=True)
    print("processing complete..")
    gt_path = "/home/maria/mcv-c6-2025-team6/data/aic19-track1-mtmc-train/train/S01/c001/gt_1.txt"
    output_txt_path = "/home/maria/mcv-c6-2025-team6/w4/output/sort/_tracking1/tracking_data.txt"
    get_hota_idf1(gt_path, output_txt_path, output_dir=os.path.dirname(output_txt_path))