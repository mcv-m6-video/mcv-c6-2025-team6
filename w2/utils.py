import os
import numpy as np
from tqdm import tqdm
def read_bboxes_from_folder(folder_path, image_width, image_height):
    bboxes_dict = {}

    # Listar todos los archivos .txt en la carpeta
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            frame_id = int(filename.split('_')[1].split('.')[0])  # Obtener el ID del frame desde el nombre del archivo (sin extensión)
            bboxes = []

            # Abrir el archivo y leer las líneas
            with open(os.path.join(folder_path, filename), 'r') as file:
                for line in file:
                    parts = line.strip().split()

                    x_center, y_center, w, h, conf = map(float, parts[1:])  # Se omite la clase

                    # Denormalizar las coordenadas
                    x_center *= image_width
                    y_center *= image_height
                    w *= image_width
                    h *= image_height

                    # Convertir de (x_center, y_center, w, h) a (x_min, y_min, x_max, y_max)
                    x_min = int(x_center - (w / 2))
                    y_min = int(y_center - (h / 2))
                    x_max = int(x_center + (w / 2))
                    y_max = int(y_center + (h / 2))

                    bboxes.append((x_min,y_min,x_max, y_max,conf))  # Agregar el bbox como tupla

            # Ordenar las cajas por la confianza (conf) en orden descendente
            bboxes.sort(key=lambda bbox: bbox[4], reverse=True)  # bbox[4] es el nivel de confianza (conf)

            # # Eliminar la confianza antes de guardar
            bboxes = [bbox[:4] for bbox in bboxes]  # Solo mantiene x, y, w, h

            # Añadir al diccionario con el frame_id como clave
            bboxes_dict[frame_id] = bboxes

    return bboxes_dict

def read_gt_and_create_dict(gt_file_path):
    gt_dict = {}

    # Leer el archivo GT
    with open(gt_file_path, 'r') as file:
        for line in file:
            # Dividir la línea en columnas
            parts = line.strip().split(',')

            # Obtener el frame_id y las coordenadas
            frame_id = int(parts[0])  # Frame ID
            x = int(float(parts[1]))       # Coordenada x
            y = int(float(parts[2]))       # Coordenada y
            x_max = int(float(parts[3]))       # x_max
            y_max = int(float(parts[4]))       # y_max
            #conf = float(parts[6])    # Confianza (no la vamos a usar, pero la extraemos)

            # Crear el bbox como tupla (x, y, w, h)
            bbox = (x, y, x_max, y_max)

            # Si el frame_id ya está en el diccionario, agregamos el bbox
            if frame_id not in gt_dict:
                gt_dict[frame_id] = []
            gt_dict[frame_id].append(bbox)

    return gt_dict

def create_mask_from_bbox(width, height, bbox, frame_id, obj_idx, dataset=None):
    """
    Create binary mask from bbox coords.
    """
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    if dataset=='gt':
        output_dir_bboxes_masks = os.path.join(output_dir, "bboxes_masks_gt")
    else:
        output_dir_bboxes_masks = os.path.join(output_dir, "bboxes_masks")
    os.makedirs(output_dir_bboxes_masks, exist_ok=True)

    mask = np.zeros((height, width), dtype=np.uint8)
    x_min, y_min, x_max, y_max = bbox

    mask[int(y_min):int(y_max), int(x_min):int(x_max)] = 255

    # cv2.imwrite(os.path.join(output_dir_bboxes_masks, f"bboxes_masks_{frame_id}_{obj_idx}.jpg"), mask)

    return mask
def binaryMaskIOU(mask1, mask2):
    mask1_area = np.count_nonzero(mask1 == 255)
    mask2_area = np.count_nonzero(mask2 == 255)
    intersection = np.count_nonzero(np.logical_and(mask1 == 255, mask2 == 255))
    union = mask1_area+mask2_area-intersection
    if union == 0: # Evitar dividir entre 0
        return 0
    iou = intersection/(union)
    return iou
def compute_ap_permuted(gt_boxes, pred_boxes):
    # Ahora solo calculamos el AP una vez, ya que las predicciones ya están ordenadas por confianza
    ap = compute_ap(gt_boxes, pred_boxes)  # Calcular el AP con las predicciones ordenadas
    return ap

def compute_ap(gt_boxes, pred_boxes):
    '''
    Extracted from Team6-2024 (https://github.com/mcv-m6-video/mcv-c6-2024-team6/blob/main/W1/task1/utils.py).
    '''
    # Initialize variables
    tp = np.zeros(len(pred_boxes))
    fp = np.zeros(len(pred_boxes))
    gt_matched = np.zeros(len(gt_boxes))

    if len(gt_boxes) == 0:
        ap = 0
    else:
        # Iterate over the predicted boxes
        for i, pred_box in enumerate(pred_boxes):
            ious = [binaryMaskIOU(pred_box, gt_box) for gt_box in gt_boxes]
            if len(ious) == 0:
                fp[i] = 1
                continue
            max_iou = max(ious)
            max_iou_idx = ious.index(max_iou)
            # print(f"IoU: {max_iou}")
            if max_iou >= 0.5 and not gt_matched[max_iou_idx]:
                tp[i] = 1
                gt_matched[max_iou_idx] = 1
            else:
                fp[i] = 1

        tp = np.cumsum(tp)
        fp = np.cumsum(fp)
        recall = tp / len(gt_boxes)
        precision = tp / (tp + fp)
        # Generate graph with the 11-point interpolated precision-recall curve
        recall_interp = np.linspace(0, 1, 11)
        precision_interp = np.zeros(11)
        for i, r in enumerate(recall_interp):
            array_precision = precision[recall >= r]
            if len(array_precision) == 0:
                precision_interp[i] = 0
            else:
                precision_interp[i] = max(precision[recall >= r])

        ap = np.mean(precision_interp)

    return ap
def calculate_mAP(aps):
    return np.mean(aps)

def process_tanda(start_idx, end_idx, gt_dict, bboxes_dict_pred, image_width, image_height):
    """Procesa una parte del diccionario y retorna los APs calculados."""
    gt_keys = list(gt_dict.keys())[start_idx:end_idx]
    pred_keys = list(bboxes_dict_pred.keys())[start_idx:end_idx]

    gt_subdict = {k: gt_dict[k] for k in gt_keys}
    pred_subdict = {k: bboxes_dict_pred[k] for k in pred_keys}

    # Crear las máscaras de GT y predicciones
    gt_masks_per_frame = {
        frame_id: [create_mask_from_bbox(image_width, image_height, bbox, frame_id, idx, dataset='gt')
                   for idx, bbox in enumerate(boxes)]
        for frame_id, boxes in gt_subdict.items()
    }

    pred_masks_per_frame = {
        frame_id: [create_mask_from_bbox(image_width, image_height, bbox, frame_id, idx)
                   for idx, bbox in enumerate(boxes)]
        for frame_id, boxes in pred_subdict.items()
    }

    aps = []
    for frame_id in tqdm(gt_subdict.keys(), desc=f"Processing frames {start_idx}-{end_idx}", position=0, leave=True):
        gt_masks = gt_masks_per_frame.get(frame_id, [])
        pred_masks = pred_masks_per_frame.get(frame_id, [])

        if len(gt_masks) > 0 or len(pred_masks) > 0:
            ap = compute_ap_permuted(gt_masks, pred_masks)
            aps.append(ap)

    return aps