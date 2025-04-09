import os
import json
import numpy as np
from SoccerNet.Evaluation.ActionSpotting import average_mAP
from dataset.frame import FPS_SN

def get_next_experiment_folder(base_output_dir, name="predict"):
    os.makedirs(base_output_dir, exist_ok=True)
    
    existing_exps = [d for d in os.listdir(base_output_dir) if d.startswith(f"_{name}")]
    
    exp_numbers = [int(d.replace(f"_{name}", "")) for d in existing_exps if d.replace(f"_{name}", "").isdigit()]

    next_exp = max(exp_numbers, default=0) + 1
    
    return os.path.join(base_output_dir, f"_{name}{next_exp}")

def compute_metrics_from_predictions(pred_dict, dataset, nms_window=5):
    """
    pred_dict: diccionario {video_name: (scores [T x C], support [T])}
    dataset: dataset usado (debe tener .videos, ._class_dict, ._labels_dir, ._stride)
    """
    detections_numpy = []

    for video, video_len, _ in dataset.videos:
        scores, support = pred_dict[video]
        support[support == 0] = 1
        scores = scores / support[:, np.newaxis]  # promedio
        pred = apply_NMS(scores, nms_window, 0.05)
        detections_numpy.append(pred)

    targets_numpy = []
    closests_numpy = []

    for video, video_len, _ in dataset.videos:
        targets = np.zeros((video_len, len(dataset._class_dict)), np.float32)
        labels = json.load(open(os.path.join(dataset._labels_dir, video, 'Labels-ball.json')))

        for annotation in labels["annotations"]:
            event = dataset._class_dict[annotation["label"]]
            frame = int(FPS_SN / dataset._stride * (int(annotation["position"]) / 1000))
            frame = min(frame, video_len - 1)
            targets[frame, event - 1] = 1

        targets_numpy.append(targets)

        closest_numpy = np.zeros(targets.shape) - 1
        for c in np.arange(targets.shape[-1]):
            indexes = np.where(targets[:, c] != 0)[0].tolist()
            if len(indexes) == 0:
                continue
            indexes.insert(0, -indexes[0])
            indexes.append(2 * closest_numpy.shape[0])
            for i in np.arange(len(indexes) - 2) + 1:
                start = max(0, (indexes[i - 1] + indexes[i]) // 2)
                stop = min(closest_numpy.shape[0], (indexes[i] + indexes[i + 1]) // 2)
                closest_numpy[start:stop, c] = targets[indexes[i], c]

        closests_numpy.append(closest_numpy)

    mAP, AP_per_class, _, _, _, _ = average_mAP(
        targets_numpy, detections_numpy, closests_numpy,
        FPS_SN / dataset._stride, deltas=np.array([1])
    )

    return mAP, AP_per_class


def apply_NMS(predictions, window, thresh=0.0):

    nf, nc = predictions.shape
    for i in range(nc):
        aux = predictions[:,i]
        aux2 = np.zeros(nf) -1
        while(np.max(aux) >= thresh):
            # Get the max remaining index and value
            max_value = np.max(aux)
            max_index = np.argmax(aux)
            # detections_NMS[max_index,i] = max_value

            nms_from = int(np.maximum(-(window/2)+max_index,0))
            nms_to = int(np.minimum(max_index+int(window/2), len(aux)))

            aux[nms_from:nms_to] = -1
            aux2[max_index] = max_value
        predictions[:,i] = aux2

    return predictions