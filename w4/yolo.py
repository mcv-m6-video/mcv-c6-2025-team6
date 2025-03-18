import os
import argparse
import torch

from ultralytics import YOLO

from utils import get_next_experiment_folder

def run_inference(source, model="yolo11n.pt", imgsz=640, conf=0.5, classes=[2, 7], save=True, save_txt=True, save_conf=True, save_frames=False, project="output", name="runs", device="cuda", stream=True):
    # Load a pretrained YOLO11n model
    model = YOLO(model)

    experiment_dir = get_next_experiment_folder(os.path.join(project, name))
    name_exp = os.path.sep.join(experiment_dir.split(os.sep)[1:])

    results = model.predict(source=source, model=model, imgsz=imgsz, conf=conf, classes=classes, save=save, save_txt=save_txt, save_conf=save_conf, project=project, name=name_exp, device=device, stream=stream)

    video_name = os.path.basename(source).split('.')[0]  # Extract video name
    output_txt_path = os.path.join(experiment_dir, f"{video_name}.txt")

    for frame_idx, result in enumerate(results):
        boxes = result.boxes.xyxy  # Boxes object for bounding box outputs
        labels = result.boxes.cls
        conf = result.boxes.conf
        detections = torch.cat((boxes, labels.unsqueeze(1), conf.unsqueeze(1)), dim=1)

        if save_frames:
            frames_dir = os.path.join(f"{experiment_dir}", "frames")
            os.makedirs(frames_dir, exist_ok=True)
            result.save(filename=f"{frames_dir}/{frame_idx+1}.jpg")  # save to disk
        
        for det in detections:
            with open(output_txt_path, 'a') as output_file:
                xtl, ytl, xbr, ybr, class_id, score = det.cpu().numpy()
                output_file.write(f"{frame_idx+1} {int(class_id)} {int(xtl)} {int(ytl)} {int(xbr)} {int(ybr)} {score}\n")
    
    print(f"Inference complete. Results saved to {output_txt_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="YOLO Inference Script")
    
    parser.add_argument("--source", type=str, required=True, help="Path to the input image or video")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Path to the YOLO model file")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for inference")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--classes", type=int, nargs='+', default=[2, 7], help="Classes to detect (space-separated list)")
    parser.add_argument("--save", type=bool, default=True, help="Save inference results (default: True)")
    parser.add_argument("--save-txt", type=bool, default=True, help="Save results as text (default: True)")
    parser.add_argument("--save-frames", type=bool, default=False, help="Save frames with same name as labels (default: False)")
    parser.add_argument("--project", type=str, default="output", help="Project folder to save results")
    parser.add_argument("--name", type=str, default="runs", help="Name of the run folder")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help="Device to run inference on")
    
    args = parser.parse_args()
    
    run_inference(
        source=args.source,
        model=args.model,
        imgsz=args.imgsz,
        conf=args.conf,
        classes=args.classes,
        save=args.save,
        save_txt=args.save_txt,
        save_frames=args.save_frames,
        project=args.project,
        name=args.name,
        device=args.device,
    )