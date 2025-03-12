import cv2
import numpy as np
from ultralytics import YOLO
import os
from w3.task_2_utils import *

# fine-tuned YOLO model
model = YOLO("C:/Users/laila/C6/week 3/task2/runs/detect/car_detection_model/weights/best.pt")  

def process_video_and_save_results(input_video_path, output_txt_file, output_frames_folder, overlap_threshold=0.9):
    cap = cv2.VideoCapture(input_video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return
    
    frame_num = 0
    if not os.path.exists(output_frames_folder):
        os.makedirs(output_frames_folder)
        
    with open(output_txt_file, 'w') as f:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            results = model(frame)  
            current_boxes = []
            print(results[0].boxes.xywh)
            for detection in results[0].boxes.xywh: 
                x, y, w, h = detection.tolist()
                current_boxes.append([int(x - w / 2), int(y - h / 2), int(w), int(h)])  # xywh to xyxy format

            filtered_boxes = filter_overlapping_boxes_based_on_iou(current_boxes, iou_threshold=overlap_threshold)

        
            for box in filtered_boxes:
                x1, y1, w, h = box
              
                cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)  
                
                
                f.write(f"{frame_num}, {x1}, {y1}, {w}, {h}\n")
            
            frame_filename = os.path.join(output_frames_folder, f"frame_{frame_num:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            
            frame_num += 1
        
    cap.release()
    print(f"Processed frames saved to {output_frames_folder} and results saved to {output_txt_file}")

if __name__ == "__main__":
    input_video_path = "data/aic19-track1-mtmc-train/train/S03/c015/vdo.avi"  
    output_txt_file = "data/aic19-track1-mtmc-train/train/S03/c015/detections.txt"
    output_frames_folder = "data/aic19-track1-mtmc-train/train/S03/c015/output_frames"
    # input_video_path = "data/AICity_data/train/S03/c010/vdo.avi"
    # output_txt_file =  "data/AICity_data/train/S03/c010/detections.txt"
    # output_frames_folder = "data/AICity_data/train/S03/c010/output_frames"
    process_video_and_save_results(input_video_path, output_txt_file, output_frames_folder)
