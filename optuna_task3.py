import optuna
from optuna.samplers import TPESampler
from utils import  load_boxes_from_txt,convert_bbox_list_to_dict,read_ground_truth,calculate_mAP,compute_ap_permuted,create_mask_from_bbox
import os 
from tqdm import tqdm
from task3 import apply_background_subtraction

def optimize_background_subtractor(trial, method='MOG', frames_path='output_frames', output_folder='output_results', trial_number=1):

    if method == 'MOG':
        history = trial.suggest_int('history', 100, 500)
        nmixtures = trial.suggest_int('nmixtures', 3, 10)
        backgroundRatio = trial.suggest_float('backgroundRatio', 0.1, 0.5)
        noiseSigma = trial.suggest_float('noiseSigma', 0.0, 50.0)
        hyperparameters = {'history': history, 'nmixtures': nmixtures, 'backgroundRatio': backgroundRatio, 'noiseSigma': noiseSigma}
    
    elif method == 'MOG2':
        history = trial.suggest_int('history', 500, 1000)
        varThreshold = trial.suggest_int('varThreshold', 10, 100)
        detectShadows = trial.suggest_categorical('detectShadows', [True, False])
        hyperparameters = {'history': history, 'varThreshold': varThreshold, 'detectShadows': detectShadows}
    
    elif method == 'LSBP':
        mc = trial.suggest_int('mc', 0, 10)
        nSamples = trial.suggest_int('nSamples', 10, 50)
        LSBPRadius = trial.suggest_int('LSBPRadius', 5, 20)
        Tlower = trial.suggest_float('Tlower', 1.0, 10.0)
        Tupper = trial.suggest_float('Tupper', 10.0, 50.0)
        hyperparameters = {'mc': mc, 'nSamples': nSamples, 'LSBPRadius': LSBPRadius, 'Tlower': Tlower, 'Tupper': Tupper}
    
    elif method == 'KNN':
        history = trial.suggest_int('history', 500, 1000)
        dist2Threshold = trial.suggest_float('dist2Threshold', 100.0, 500.0)
        detectShadows = trial.suggest_categorical('detectShadows', [True, False])
        hyperparameters = {'history': history, 'dist2Threshold': dist2Threshold, 'detectShadows': detectShadows}
    
    elif method == 'GMG':
        initializationFrames = trial.suggest_int('initializationFrames', 100, 200)
        decisionThreshold = trial.suggest_float('decisionThreshold', 0.5, 1.0)
        hyperparameters = {'initializationFrames': initializationFrames, 'decisionThreshold': decisionThreshold}
    
    elif method == 'GSOC':
        mc = trial.suggest_int('mc', 0, 10)
        nSamples = trial.suggest_int('nSamples', 10, 50)
        replaceRate = trial.suggest_float('replaceRate', 0.001, 0.01)
        propagationRate = trial.suggest_float('propagationRate', 0.001, 0.01)
        hyperparameters = {'mc': mc, 'nSamples': nSamples, 'replaceRate': replaceRate, 'propagationRate': propagationRate}
    
    elif method == 'CNT':
        minPixelStability = trial.suggest_int('minPixelStability', 5, 50)
        useHistory = trial.suggest_categorical('useHistory', [True, False])
        maxPixelStability = trial.suggest_int('maxPixelStability', 300, 900)
        isParallel = trial.suggest_categorical('isParallel', [True, False])
        hyperparameters = {'minPixelStability': minPixelStability, 'useHistory': useHistory, 'maxPixelStability': maxPixelStability, 'isParallel': isParallel}
    
    elif method == 'Lobster':
        hyperparameters = {}

    else:
        raise ValueError(f"Unsupported method: {method}")

    apply_background_subtraction(method, frames_path, output_folder, trial_number, **hyperparameters)

    results_file = "output/results.txt"
    with open(results_file, "a") as f:
   
        xml_file = 'dataset/ai_challenge_s03_c010-full_annotation.xml'
        classes = ['car']  # The other class is bike
        gt_boxes = read_ground_truth(xml_file, classes, len(os.listdir(frames_path)))
        gt_boxes_per_frame = convert_bbox_list_to_dict(gt_boxes)
        
        pred_boxes_per_frame = load_boxes_from_txt(os.path.join(output_folder, f"trial_{trial_number}_{method}", '_bbox_results.txt'))
        
       
        image_width, image_height = 1920, 1080
        gt_masks_per_frame = {frame_id: [create_mask_from_bbox(image_width, image_height, bbox, frame_id, idx, dataset='gt') for idx, bbox in enumerate(boxes)] 
                            for frame_id, boxes in gt_boxes_per_frame.items()}
            
        image_width, image_height = 1920, 1080
        pred_masks_per_frame = {frame_id: [create_mask_from_bbox(image_width, image_height, bbox, frame_id, idx) for idx, bbox in enumerate(boxes)] 
                                    for frame_id, boxes in pred_boxes_per_frame.items()}

        aps = []
        start_frame = int(len(gt_masks_per_frame)*0.25)
           
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
                    
        mAP = calculate_mAP(aps)
        f.write(f"mAP = {mAP},{method}\n")

    return mAP


def main_optimization(method='MOG', video_path='dataset/AICity_data/AICity_data/train/S03/c010/vdo.avi', frames_path='output_frames', output_folder='output_results_optuna', trial_number=1):
    study = optuna.create_study(direction='maximize', sampler=TPESampler())
    
    study.optimize(lambda trial: optimize_background_subtractor(trial, method, video_path, frames_path, output_folder, trial_number), n_trials=5)

    with open("optuna_results.txt", "a") as result_file:
        result_file.write(f"Method: {method}\n")
        result_file.write(f"Best hyperparameters found: {study.best_params}\n")
        result_file.write(f"Best mAP score: {study.best_value}\n")
        result_file.write("-" * 50 + "\n")

    print(f"Best hyperparameters found for {method}: {study.best_params}")
    print(f"Best mAP score for {method}: {study.best_value}")


if __name__ == "__main__":
    methods = ['MOG', 'MOG2', 'LSBP', 'KNN', 'GMG', 'GSOC', 'CNT']
    
    for method in methods:
        print(f"Optimizing hyperparameters for: {method}")
        main_optimization(method=method, frames_path='output_frames', output_folder='output_results_optuna', trial_number=1)
