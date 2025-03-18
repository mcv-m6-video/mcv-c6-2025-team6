import os
import cv2
import argparse

from deep_sort_realtime.deepsort_tracker import DeepSort

from utils import get_next_experiment_folder
from metrics import get_hota_idf1



def run_deepsort(input_video_path, detections_file, output_dir):
    # Initialize DeepSORT tracker
    tracker = DeepSort(max_age=5, 
                    nms_max_overlap=0.7, 
                    max_cosine_distance=0.4, 
                    max_iou_distance=0.7,
                    embedder="clip_ViT-B/32")

    # Open video stream
    cap = cv2.VideoCapture(input_video_path)

    # Path to your detections file
    detections_dict = {}

    # Leer las detecciones y almacenarlas en un diccionario por frame_id
    with open(detections_file, "r") as f:
        for line in f.readlines():
            frame_id, class_id, xtl, ytl, xbr, ybr, score = map(float, line.strip().split())
            frame_id = int(frame_id)  # Asegurarse de que frame_id sea un entero
            xtl, ytl, xbr, ybr = map(int, [xtl, ytl, xbr, ybr])  # Coordenadas en píxeles
            confidence = score  # Usamos el score como confianza

            # Asegurarse de que exista una lista para ese frame_id
            if frame_id not in detections_dict:
                detections_dict[frame_id] = []
            
            # Añadir la detección con formato adecuado para DeepSORT
            detections_dict[frame_id].append(([xtl, ytl, xbr-xtl, ybr-ytl], confidence, "vehicle"))  # "vehicle" es un ejemplo de clase

    # Obtener las propiedades del video
    frame_idx = 0
    ret, frame = cap.read()
    height, width, _ = frame.shape  # Obtener el tamaño del frame (alto, ancho)

    # Configurar el VideoWriter para guardar el video
    experiment_dir = get_next_experiment_folder(output_dir, name="tracking")
    os.makedirs(experiment_dir, exist_ok=True)
    output_video_path = f"{experiment_dir}/vdo.avi"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Usamos el codec XVID
    video_writer = cv2.VideoWriter(output_video_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # frame_idx += 1  # Indice del frame actual

        # Obtener las detecciones para el frame actual
        detections = detections_dict.get(frame_idx, [])
        # Actualizar el tracker con las detecciones
        tracked_objects = tracker.update_tracks(detections, frame=frame)

        # Dibujar los resultados de tracking
        for obj in tracked_objects:
            if not obj.is_confirmed():
                continue

            track_id = obj.track_id
            ltrb = obj.to_ltrb()  # [xmin, ymin, xmax, ymax]

            xtl, ytl, xbr, ybr = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])

            cv2.rectangle(frame, (xtl, ytl), (xbr, ybr), (255, 255, 255), 2)

            # Medir el tamaño del texto
            label = f"ID {track_id}"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            # Dibujar rectángulo de fondo para el texto
            text_x1, text_y1 = xtl, ytl - text_size[1] - 2
            text_x2, text_y2 = xtl + text_size[0] + 2, ytl
            cv2.rectangle(frame, (text_x1, text_y1), (text_x2, text_y2), (255,255,255), -1)
            cv2.putText(frame, f'ID {track_id}', (xtl + 5, ytl - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

            output_txt_path = os.path.join(experiment_dir, "tracks.txt")
            with open(output_txt_path, 'a') as out_file:
                line = f"{int(frame_idx)}, {int(track_id)}, {int(xtl)}, {int(ytl)}, {int(xbr)-int(xtl)}, {int(ybr)-int(ytl)}, 1, -1, -1, -1\n"
                out_file.write(line)
        
        # Escribir el frame procesado en el video
        video_writer.write(frame)
        frame_idx += 1  # Indice del frame actual

    # Liberar recursos y cerrar el video
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()

    print(f"Video processed and saved to {output_video_path}")
    
    return output_txt_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DeepSORT tracking on a video and save results.")
    parser.add_argument('--input_video_path', type=str, help='Path to the input video file')
    parser.add_argument('--gt', type=str, help='Path to the gt file')
    parser.add_argument('--detections_file', type=str, help='Path to the detections file')
    parser.add_argument('--output_dir', type=str, help='Directory to save the output results')
    args = parser.parse_args()

    # Run the DeepSORT tracking and get the output tracks file path
    output_txt_path = run_deepsort(args.input_video_path, args.detections_file, args.output_dir)

    # Now calculate HOTA and IDF1 metrics
    if args.gt is not None:
        get_hota_idf1(gt_path=args.gt, det_path=output_txt_path, output_dir=os.path.dirname(output_txt_path))

