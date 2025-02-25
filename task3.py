from utils import extract_frames, load_boxes_from_txt,convert_bbox_list_to_dict,read_ground_truth,calculate_mAP,compute_ap,compute_ap_permuted,create_mask_from_bbox
import cv2 
import os 
from tqdm import tqdm

def filter_mask(mask):
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))  
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))  
    
    # median blur to reduce salt-and-pepper noise
    mask = cv2.medianBlur(mask, 3)  

    # opening followed by closing to remove small noise and close holes
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=2) 
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel_close, iterations=3)  

    return closing

class BackgroundSubtractorMOG:
    def __init__(self, history=500, nmixtures=5, backgroundRatio=0.3, noiseSigma=0):
        self.history = history
        self.nmixtures = nmixtures
        self.backgroundRatio = backgroundRatio
        self.noiseSigma = noiseSigma
        self.model = cv2.bgsegm.createBackgroundSubtractorMOG(history, nmixtures, backgroundRatio, noiseSigma)

    def apply(self, frame):
        return self.model.apply(frame)


class BackgroundSubtractorMOG2:
    def __init__(self, history=500, varThreshold=16, detectShadows=True):
        self.history = history
        self.varThreshold = varThreshold
        self.detectShadows = detectShadows
        self.model = cv2.createBackgroundSubtractorMOG2(history, varThreshold, detectShadows)

    def apply(self, frame):
        return self.model.apply(frame)


class BackgroundSubtractorLSBP:
    def __init__(self, mc=0, nSamples=20, LSBPRadius=16, Tlower=2.0, Tupper=32.0):
        self.mc = mc
        self.nSamples = nSamples
        self.LSBPRadius = LSBPRadius
        self.Tlower = Tlower
        self.Tupper = Tupper
        self.model = cv2.bgsegm.createBackgroundSubtractorLSBP(
            mc=self.mc, 
            nSamples=self.nSamples, 
            LSBPRadius=self.LSBPRadius,
            Tlower=self.Tlower, 
            Tupper=self.Tupper
        )

    def apply(self, frame):
        return self.model.apply(frame)


class BackgroundSubtractorKNN:
    def __init__(self, history=500, dist2Threshold=400.0, detectShadows=True):
        self.history = history
        self.dist2Threshold = dist2Threshold
        self.detectShadows = detectShadows
        self.model = cv2.createBackgroundSubtractorKNN(history, dist2Threshold, detectShadows)

    def apply(self, frame):
        return self.model.apply(frame)


class BackgroundSubtractorGMG:
    def __init__(self, initializationFrames=120, decisionThreshold=0.8):
        self.initializationFrames = initializationFrames
        self.decisionThreshold = decisionThreshold
        self.model = cv2.bgsegm.createBackgroundSubtractorGMG(initializationFrames, decisionThreshold)

    def apply(self, frame):
        return self.model.apply(frame)


class BackgroundSubtractorGSOC:
    def __init__(self, mc=0, nSamples=20, replaceRate=0.003, propagationRate=0.01):
        self.mc = mc
        self.nSamples = nSamples
        self.replaceRate = replaceRate
        self.propagationRate = propagationRate
        self.model = cv2.bgsegm.createBackgroundSubtractorGSOC(
            mc=self.mc, 
            nSamples=self.nSamples,
            replaceRate=self.replaceRate, 
            propagationRate=self.propagationRate
        )

    def apply(self, frame):
        return self.model.apply(frame)


class BackgroundSubtractorCNT:
    def __init__(self, minPixelStability=15, useHistory=True, maxPixelStability=15*60, isParallel=True):
        self.minPixelStability = minPixelStability
        self.useHistory = useHistory
        self.maxPixelStability = maxPixelStability
        self.isParallel = isParallel
        self.model = cv2.bgsegm.createBackgroundSubtractorCNT(
            minPixelStability=self.minPixelStability, 
            useHistory=self.useHistory, 
            maxPixelStability=self.maxPixelStability, 
            isParallel=self.isParallel
        )

    def apply(self, frame):
        return self.model.apply(frame)


class BackgroundSubtractorFactory:
    @staticmethod
    def create(method, **kwargs):
        if method == 'MOG':
            return BackgroundSubtractorMOG(**kwargs)
        elif method == 'MOG2':
            return BackgroundSubtractorMOG2(**kwargs)
        elif method == 'LSBP':
            return BackgroundSubtractorLSBP(**kwargs)
        elif method == 'KNN':
            return BackgroundSubtractorKNN(**kwargs)
        elif method == 'GMG':
            return BackgroundSubtractorGMG(**kwargs)
        elif method == 'GSOC':
            return BackgroundSubtractorGSOC(**kwargs)
        elif method == 'CNT':
            return BackgroundSubtractorCNT(**kwargs)
        elif method == "Lobster":
            pass
        else:
            raise ValueError(f"Unsupported method: {method}")

def save_bounding_boxes(bbox_info, output_path):
    """
    bbox_info is a list of lists where each list contains bounding boxes for a frame.
    Each bounding box is represented as [xtl, ytl, xbr, ybr].
    """
    with open(output_path, 'w') as f:
        for frame_id, frame_bboxes in enumerate(bbox_info):
            for bbox in frame_bboxes:
                x, y, w, h = bbox
                f.write(f"{frame_id+1},{x},{y},{w},{h}\n")

def apply_background_subtraction(method, frame_folder, output_folder, trial_number, **hyperparameters):
    
    fgbg = BackgroundSubtractorFactory.create(method, **hyperparameters)

    output_path = f"{output_folder}/trial_{trial_number}_{method}"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    bounding_boxes = []
    frame_files = [f for f in os.listdir(frame_folder) if f.endswith('.jpg')]
    frame_files.sort()  # to ensure consistent order
    n_frames = len(frame_files)
    bbox_info = [[] for _ in range(n_frames)]

    for idx, frame_file in enumerate(frame_files):
        frame_path = os.path.join(frame_folder, frame_file)
        frame = cv2.imread(frame_path)
        if frame is None:
            continue

        fg_mask = fgbg.apply(frame)
        fg_mask_clean = filter_mask(fg_mask)  

        mask_filename = f"{output_path}/frame_mask{idx + 1}.jpg"
        cv2.imwrite(mask_filename, fg_mask)
        print(f"Saved: {mask_filename}")

        contours, _ = cv2.findContours(fg_mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) > 500:  # area threshold to ignore noise
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                bounding_boxes.append((x, y, w, h))
                bbox_info[idx].append([x, y, w, h])

        output_filename = f"{output_path}/frame_{idx + 1}.jpg"
        cv2.imwrite(output_filename, frame)
        print(f"Saved: {output_filename}")
    bbox_results_path = os.path.join(output_path,'_bbox_results.txt')
    save_bounding_boxes(bbox_info, bbox_results_path)
    print(f"Saved bounding boxes to: {bbox_results_path}")

    return bounding_boxes

        

def main(method='MOG',video_path='dataset/AICity_data/AICity_data/train/S03/c010/vdo.avi', frames_path='output_frames', output_folder='output_results', trial_number=1):
    
    #extract_frames(video_path,frames_path)
    results_file = "output/results.txt"
    os.makedirs("output", exist_ok=True)

    with open(results_file, "a") as f:
            
            xml_file = 'dataset/ai_challenge_s03_c010-full_annotation.xml'
            classes = ['car'] # The other class is bike
            path = 'output_frames/'
            frames_list = sorted(os.listdir(path))
            n_frames = len(frames_list)
            gt_boxes = read_ground_truth(xml_file, classes, n_frames)
            gt_boxes_per_frame = convert_bbox_list_to_dict(gt_boxes)

            # apply background subtraction and save the frames with bounding boxes
            apply_background_subtraction(method, frames_path, output_folder, trial_number)
        
            pred_boxes_per_frame = load_boxes_from_txt(os.path.join(f"{output_folder}/trial_{trial_number}_{method}", '_bbox_results.txt'))

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
            
            mAP = calculate_mAP(aps)
            print(f"mAP = {mAP}\n")
            f.write(f"mAP = {mAP},{method}\n")
    

#if __name__ == "__main__":

    #main(method='Lobster', frames_path='output_frames', output_folder='output_results', trial_number=8)