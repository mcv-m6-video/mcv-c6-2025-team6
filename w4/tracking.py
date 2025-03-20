import torch
import cv2
import numpy as np
from pathlib import Path
import os

from boxmot import ByteTrack
from boxmot import BotSort
from ultralytics import YOLO 

print(torch.cuda.is_available())


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
detector = YOLO("yolo11n.pt").to(device)  

# Initialize the tracker
tracker = ByteTrack()
# tracker = BotSort(
#     reid_weights=Path('osnet_x0_25_msmt17.pt'),  # Path to ReID model
#     device=device,  # Use CPU or GPU for inference
#     half=False
# )

experiment_dir = "C:/Users/laila/C6/week 3/task2/aic19-track1-mtmc-train/train/S04"  
output_dir="C:/Users/laila/boxmot/examples/det/byteTrack_results"


for cam_id in range(16, 41):
    cam_str = f"c{cam_id:03d}"  # 'c016', 'c017', ..., 'c040'
    
    input_video_path = os.path.join(experiment_dir, cam_str, "vdo.avi")
    output_video_path = os.path.join(output_dir, f"output_{cam_str}.avi")
    output_txt_path = os.path.join(output_dir, f"tracks_{cam_str}.txt")
    
    print(f"Processing video for {cam_str}:")
    print(f"Input path: {input_video_path}")
    print(f"Output video path: {output_video_path}")
    print(f"Output text path: {output_txt_path}")

    vid = cv2.VideoCapture(input_video_path)

    frame_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    frame_idx = 1

    # filtering small objects
    size_threshold = 5000

    while True:
        ret, frame = vid.read()

        if not ret:
            break

        results = detector(frame) 

        dets = []
        for result in results:
            for box in result.boxes:
                xyxy = box.xyxy.cpu().numpy()[0]  
                conf = box.conf.cpu().numpy()[0]  
                cls = box.cls.cpu().numpy()[0]  

                # Keep detections of class 2 'car' and class 7 'truck' (YOLO class IDs)
                if cls in [2, 7]:
                    # Calculate the area of the bounding box
                    area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
                    # Keep detections with area above the threshold
                    if area >= size_threshold:
                        dets.append([*xyxy, conf, cls])

        # Convert detections to numpy array (N X (x, y, x, y, conf, cls))
        dets = np.array(dets)

        # Update the tracker
        res = tracker.update(dets, frame)  # --> M X (x, y, x, y, id, conf, cls, ind)
        print(res)

        with open(output_txt_path, 'a') as out_file:
            for track in res:
                xtl, ytl, xbr, ybr, track_id, conf, cls, ind = track
                width = int(xbr - xtl)
                height = int(ybr - ytl)
                line = f"{int(frame_idx)}, {int(track_id)}, {int(xtl)}, {int(ytl)}, {width}, {height}, 1, -1, -1, -1\n"
                out_file.write(line)

        tracker.plot_results(frame, show_trajectories=True)

        out.write(frame)

        cv2.imshow('BoXMOT + YOLO', frame)

        frame_idx += 1

        # Exit on 'q' key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    vid.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Output video saved to: {output_video_path}")
    print(f"Tracking results saved to: {output_txt_path}")