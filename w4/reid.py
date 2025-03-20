import os
import cv2
import numpy as np
import torch
import pickle
import random
from PIL import Image
import faiss
from torchvision import transforms
from fastreid.modeling import build_model
from fastreid.config import get_cfg
from sklearn.metrics.pairwise import cosine_similarity
# from metrics import get_hota_idf1
from test_metric import calculate_metrics_taking_only_GTobject_into_account, parse_tracking_file


def cossim(input1, input2):
    return cosine_similarity(input1.reshape(1, -1), input2.reshape(1, -1))

def read_detections(txt_file):
    detections = []
    with open(txt_file, 'r') as f:
        for line in f.readlines():
            frame_id, track_id, x, y, w, h, _, _, _, _ = map(int, line.strip().split(','))
            detections.append((frame_id, track_id, (x, y, w, h)))
    return detections

def crop_object(frame, bbox):
    xtl, ytl, w, h = bbox
    xbr, ybr = xtl + w, ytl + h
    xtl, ytl = max(0, xtl), max(0, ytl)
    xbr, ybr = min(xbr, frame.shape[1] - 1), min(ybr, frame.shape[0] - 1)
    if (xbr - xtl) == 0 or (ybr - ytl) == 0:
        return None
    cropped_pil = Image.fromarray(frame[ytl:ybr, xtl:xbr])
    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(cropped_pil)

def extract_features(model, images):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_tensor = torch.stack(images).to(device)
    with torch.no_grad():
        batch_features = model(batch_tensor)
    return batch_features.cpu().numpy()

def create_database(data_folder, batch_size=128):
    video_paths, txt_files = {}, {}

    # Automatically detect video and annotation files
    for file in os.listdir(data_folder):
        if file.endswith(".avi") or file.endswith(".mp4"):
            cam_id = file.split(".")[0]  # Extract "S01_c001" from "S01_c001.avi"
            video_paths[cam_id] = os.path.join(data_folder, file)
        elif file.endswith(".txt"):
            cam_id = file.split(".")[0]
            txt_files[cam_id] = os.path.join(data_folder, file)

    cap_list = {cam_id: cv2.VideoCapture(video_paths[cam_id]) for cam_id in video_paths}
    faiss_indexes = {cam_id: faiss.IndexFlatIP(2048) for cam_id in txt_files}
    track_info = {cam_id: [] for cam_id in txt_files}

    if torch.cuda.is_available():
        for cam_id in faiss_indexes:
            faiss_indexes[cam_id] = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, faiss_indexes[cam_id])

    cfg = get_cfg()
    cfg.merge_from_file("fast-reid/configs/VERIWild/bagtricks_R50-ibn.yml")
    model = build_model(cfg).to(torch.device("cuda" if torch.cuda.is_available() else "cpu")).eval()
    all_detections = []
    for cam_id, txt_file in txt_files.items():
        detections = read_detections(txt_file)
        all_detections.append(detections)
        batch_images, batch_track_info = [], []
        print(f"Processing camera {cam_id}")
        for frame_id, track_id, bbox in detections:
            cap = cap_list[cam_id]
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id - 1)
            ret, frame = cap.read()
            if not ret:
                continue

            cropped_img = crop_object(frame, bbox)
            if cropped_img is None:
                continue

            batch_images.append(cropped_img)
            batch_track_info.append((track_id, frame_id))

            if len(batch_images) >= batch_size:
                batch_features = extract_features(model, batch_images)
                faiss_indexes[cam_id].add(batch_features)
                track_info[cam_id].extend(batch_track_info)
                batch_images, batch_track_info = [], []

        if batch_images:
            batch_features = extract_features(model, batch_images)
            faiss_indexes[cam_id].add(batch_features)
            track_info[cam_id].extend(batch_track_info)

    for cap in cap_list.values():
        cap.release()

    return faiss_indexes, track_info, all_detections

def match_across_cameras(faiss_indexes, track_info, threshold=0.94, frame_range=10):
    matches, cameras = [], list(faiss_indexes.keys())
    all_matches = []
    camera_pairs = []
    for i in range(len(cameras)):
        for j in range(i + 1, len(cameras)):
            cam1, cam2 = cameras[i], cameras[j]
            print(f"Processing matches between cameras {cam1} and {cam2}, window_size: {frame_range}, threshold: {threshold}")
            index1, index2 = faiss_indexes[cam1], faiss_indexes[cam2]
            embeddings1 = np.vstack([index1.reconstruct(n) for n in range(index1.ntotal)])
            embeddings2 = np.vstack([index2.reconstruct(n) for n in range(index2.ntotal)])

            for idx1, embedding1 in enumerate(embeddings1):
                track1, frame1 = track_info[cam1][idx1]
                start_frame, end_frame = max(1, frame1 - frame_range), frame1 + frame_range

                best_match, best_dist = -1, -1
                for idx2, (embedding2, (track2, frame2)) in enumerate(zip(embeddings2, track_info[cam2])):
                    if start_frame <= frame2 <= end_frame:
                        dist = cossim(embedding1, embedding2)[0]
                        if dist > best_dist:
                            best_match, best_dist = idx2, dist

                if best_match != -1 and best_dist > threshold:
                    # print(f"Accepting match between frame {frame1} and track {track1} (cam1) with (frame {track_info[cam2][best_match][1]}, track {track_info[cam2][best_match][0]}) (cam2) "
                    #       f"with dist {best_dist} > {threshold} (threshold)")
                    matches.append([frame1, track1, track_info[cam2][best_match][1], track_info[cam2][best_match][0], best_dist[0]])
            camera_pairs.append((cam1, cam2))
            all_matches.append(np.array(matches))
            matches = []
    
    return all_matches, camera_pairs

def load_tracks(file_path):
    return np.loadtxt(file_path, delimiter=',', dtype=int)

def save_faiss_data(faiss_indexes, track_info, faiss_index_path, track_info_path):
    # Guardar los índices de Faiss
    for cam_id, index in faiss_indexes.items():
        # Mover el índice de la GPU a la CPU si está en la GPU
        if hasattr(index, 'is_trained') and isinstance(index, faiss.Index):
            # Si el índice está en la GPU, lo movemos a la CPU
            try:
                index = faiss.index_gpu_to_cpu(index)
            except Exception as e:
                print(f"Error moviendo el índice a la CPU: {e}")
        
        # Guardar el índice en un archivo
        faiss.write_index(index, f"{faiss_index_path}_{cam_id}.index")

    # Guardar la información de las pistas (track info)
    with open(track_info_path, 'wb') as f:
        pickle.dump(track_info, f)

def load_faiss_data(faiss_index_path, track_info_path):
    # Cargar la información de las pistas (track info)
    with open(track_info_path, 'rb') as f:
        track_info = pickle.load(f)
    
    cam_ids = track_info.keys()
    
    faiss_indexes = {}
    # Cargar los índices de Faiss
    for cam_id in cam_ids:
        index = faiss.read_index(f"{faiss_index_path}_{cam_id}.index")
        
        # Mover los índices a la GPU si es necesario
        if torch.cuda.is_available():
            index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index)
        
        faiss_indexes[cam_id] = index

    return faiss_indexes, track_info, cam_ids


def build_track_matrix(camera_pairs, match_arrays, num_cameras):
        mega_array = []  # Lista principal donde cada fila es una lista con listas internas por cámara
        camera_names = {}
        for (cam_a, cam_b), matches in zip(camera_pairs, match_arrays):
            idx_a, idx_b = int(cam_a[-3:]) - 1, int(cam_b[-3:]) - 1  # Convertir nombres de cámaras a índices
            
            if cam_a not in camera_names.keys():
                camera_names[idx_a] = cam_a
            if cam_b not in camera_names.keys():
                camera_names[idx_b] = cam_b
            
            for track_a, track_b in matches:
                row_a, row_b = None, None

                # Buscar si los tracks ya están en alguna fila
                for i, row in enumerate(mega_array):
                    if track_a in row[idx_a]:
                        row_a = i
                    if track_b in row[idx_b]:
                        row_b = i

                if row_a is None and row_b is None:
                    # Si ninguno de los tracks ha sido matcheado, añadimos una nueva fila
                    new_row = [[] for _ in range(num_cameras)]
                    new_row[idx_a].append(int(track_a))
                    new_row[idx_b].append(int(track_b))
                    mega_array.append(new_row)

                elif row_a is not None and row_b is None:
                    # Si el track_a ya está matcheado pero el track_b no, añadimos track_b en la fila de track_a
                    # Tengo que verificar si track_b coincide en frame con alguno de los que ya existen en esa fila
                    # Si no coincide, lo agrego ahi
                    # Si coincide -- verificar el score de match entre los elementos b de esa fila y el track a 
                    mega_array[row_a][idx_b].append(int(track_b))

                elif row_a is None and row_b is not None:
                    # Si el track_b ya está matcheado pero el track_a no, añadimos track_a en la fila de track_b
                    mega_array[row_b][idx_a].append(int(track_a))

                elif row_a != row_b:
                    # Si ambos están matcheados pero en diferentes filas, fusionamos las filas
                    mega_array[row_a] = [mega_array[row_a][i] + mega_array[row_b][i] for i in range(num_cameras)]
                    del mega_array[row_b]  # Eliminar la fila duplicada

        return mega_array, camera_names

# --------------------

def run_reid(window_size, threshold, data_folder="/home/maria/mcv-c6-2025-team6/data/aic19-track1-mtmc-train/train/complete_videos/S01"):

    # faiss_indexes, track_info, all_detections = create_database(data_folder)
    # save_faiss_data(faiss_indexes, track_info, "faiss_index", "track_info.pkl")

    faiss_indexes, track_info, cam_ids = load_faiss_data("faiss_index", "track_info.pkl")
    unified_ids, camera_pairs = match_across_cameras(faiss_indexes, track_info, threshold=threshold, frame_range=window_size)

    for idx, match_array in enumerate(unified_ids):
        np.save(f"matches_{idx}.npy", match_array)
    np.save(f"num_match_arrays.npy", idx)

    np.save(f"camera_pairs", np.array(camera_pairs))

    all_tracks = {}
    for cam_id in track_info.keys():
        cam_data = load_tracks(os.path.join(data_folder, f"{cam_id}.txt"))
        track_ids = cam_data[:, 1]
        all_tracks[cam_id] = {"data": cam_data, "original_tracks": track_ids, "new_col": np.full_like(track_ids, -2)}


    camera_pairs = np.load("camera_pairs.npy")

    num_match_arrays = np.load("num_match_arrays.npy")
    matches_list = []
    for idx in range(num_match_arrays+1):
        if os.path.exists(f"matches_{idx}.npy"):
            m = np.load(f"matches_{idx}.npy")
            if m.ndim >= 2:
                matches_list.append(np.unique(m[:, [1, 3]], axis=0))
            else:
                matches_list.append([])
        else:
            matches_list.append([])

    # matches_list = []
    # for i in range(len(unified_ids)):
    #     matches_list.append(np.unique(unified_ids[:, [1, 3]], axis=0))
    # print(matches_list)

    num_cameras = len(track_info)
    all_camera_matches, camera_names = build_track_matrix(camera_pairs, matches_list, num_cameras=num_cameras)


    # cam_ids = list(camera_names.values())  # O cualquier otra lista de cámaras que tengas
    # faiss_indexes, track_info = load_faiss_data("faiss_index", "track_info.pkl", cam_ids)

    # for row in all_camera_matches:
    #     print(row)

    # track_info = {'S01_c003': [(3, 1), (2, 1), (1, 1), (5, 2), (4, 2), (3, 2), (2, 2), (1, 2), (3, 3), (2, 3), (1, 3), (4, 4), (3, 4), (2, 4), (1, 4), (4, 5), (2, 5), (1, 5), (2, 6), (1, 6), (3, 7), (2, 7), (1, 7), (3, 8), (2, 8), (1, 8), (3, 9), (2, 9), (1, 9), (3, 10), (2, 10), (8, 11), (3, 11), (2, 11), (3, 12), (2, 12), (8, 13), (3, 13), (2, 13), (7, 14), (3, 14), (2, 14), (5, 15), (3, 15), (2, 15), (3, 16), (2, 16), (3, 17), (2, 17), (3, 18), (2, 18), (3, 19), (2, 19), (10, 20), (3, 20), (2, 20), (3, 21), (2, 21), (10, 22), (5, 22), (3, 22), (2, 22), (3, 23), (3, 24), (5, 25), (3, 25), (12, 26), (11, 26), (5, 26), (13, 27), (5, 27), (5, 28), (10, 29), (5, 29), (14, 30), (10, 30), (5, 30), (11, 31), (5, 31), (11, 32), (5, 32), (11, 33), (5, 33), (11, 34), (5, 34), (15, 35), (11, 35), (5, 35), (15, 36), (5, 36), (15, 37), (14, 37), (5, 37), (15, 38), (11, 38), (5, 38), (15, 39), (14, 39), (5, 39), (15, 40), (14, 40), (10, 40), (5, 40), (15, 41), (14, 41), (11, 41), (10, 41), (5, 41), (15, 42), (14, 42), (5, 42), (15, 43), (14, 43), (10, 43), (5, 43), (15, 44), (14, 44), (5, 44), (15, 45), (14, 45), (5, 45), (15, 46), (14, 46), (11, 46), (5, 46), (15, 47), (14, 47), (5, 47), (14, 48), (11, 48), (5, 48), (17, 49), (14, 49), (11, 49), (5, 49), (14, 50), (15, 51), (14, 51), (11, 51), (14, 52), (5, 52), (14, 53), (5, 53), (14, 54), (5, 54), (14, 55), (5, 55), (14, 56), (14, 57), (14, 58), (11, 58), (16, 59), (14, 59), (14, 60), (14, 61), (14, 62), (14, 63), (11, 63), (14, 64), (14, 65), (14, 66), (14, 67), (14, 68), (14, 69), (14, 70), (14, 71), (11, 71), (14, 72), (14, 73), (11, 73), (14, 74), (14, 75), (14, 76), (14, 77), (21, 78), (14, 78), (22, 79), (14, 79), (22, 80), (21, 80), (14, 80), (22, 81), (14, 81), (22, 82), (22, 83), (22, 84), (24, 88), (25, 89), (11, 89), (25, 90), (25, 91), (11, 91), (26, 92), (25, 92), (25, 93), (26, 94), (25, 94), (26, 95), (25, 95), (25, 96), (25, 97), (26, 98), (25, 98), (26, 99), (25, 99), (25, 100), (11, 100), (25, 101), (26, 102), (25, 102), (26, 103), (25, 103), (26, 104), (25, 104), (25, 105), (26, 107), (28, 108), (26, 108), (28, 109), (26, 109), (28, 110), (26, 110), (28, 111), (28, 112), (28, 113), (26, 113), (11, 113), (28, 114), (26, 114), (11, 114), (28, 115), (26, 115), (29, 116), (28, 116), (26, 116), (28, 117), (26, 117), (28, 118), (26, 118), (28, 119), (26, 119), (28, 120), (26, 120), (28, 121), (28, 122), (28, 123), (28, 124), (28, 125), (28, 126), (26, 126), (28, 127), (26, 127), (28, 128), (26, 128), (28, 129), (26, 129), (28, 130), (26, 130), (29, 131), (28, 131), (26, 131), (28, 132), (26, 132), (29, 133), (28, 133), (29, 134), (28, 134), (32, 135), (28, 135), (28, 136), (32, 137), (29, 137), (28, 137), (29, 138), (28, 138), (32, 139), (28, 139), (32, 140), (32, 142), (29, 144), (32, 145), (32, 146), (33, 148), (33, 149), (32, 149), (29, 149), (33, 150), (32, 150), (34, 151), (33, 151), (32, 151), (34, 152), (33, 152), (32, 152), (34, 153), (33, 153), (32, 153), (34, 154), (33, 154), (32, 154), (34, 155), (33, 155), (32, 155), (35, 156), (34, 156), (33, 156), (32, 156), (33, 157), (32, 157), (33, 158), (32, 158), (29, 158), (33, 159), (32, 159), (29, 159), (33, 160), (32, 160), (29, 160), (34, 161), (33, 161), (32, 161), (34, 162), (33, 162), (32, 162), (34, 163), (33, 163), (32, 163), (34, 164), (33, 164), (32, 164), (36, 165), (34, 165), (33, 165), (32, 165), (36, 166), (34, 166), (33, 166), (32, 166)], 'S01_c001': [(5, 1), (4, 1), (3, 1), (2, 1), (1, 1), (4, 2), (3, 2), (2, 2), (1, 2), (3, 3), (2, 3), (1, 3), (3, 4), (2, 4), (1, 4), (3, 5), (2, 5), (1, 5), (3, 6), (2, 6), (1, 6), (3, 7), (2, 7), (1, 7), (3, 8), (2, 8), (1, 8), (3, 9), (2, 9), (1, 9), (3, 10), (2, 10), (1, 10), (3, 11), (2, 11), (1, 11), (3, 12), (2, 12), (1, 12), (3, 13), (2, 13), (1, 13), (3, 14), (2, 14), (1, 14), (3, 15), (2, 15), (8, 17), (11, 19), (5, 19), (11, 20), (11, 21), (11, 22), (11, 23), (11, 24), (11, 25), (11, 26), (11, 27), (11, 28), (11, 29), (11, 30), (11, 31), (11, 32), (11, 33), (11, 34), (11, 35), (13, 60), (13, 61), (5, 61), (13, 62), (13, 63), (13, 64), (13, 65), (13, 66), (13, 67), (13, 68), (16, 89), (16, 90), (16, 91), (16, 92), (16, 93), (16, 94), (18, 111), (18, 112), (18, 113), (18, 114), (18, 115), (18, 116), (18, 117), (18, 118), (18, 119), (18, 120), (18, 121), (18, 122), (18, 123), (18, 124), (20, 138), (20, 139), (20, 140), (20, 141), (20, 142), (20, 143), (20, 144), (20, 145), (22, 167)], 'S01_c002': [(4, 1), (3, 1), (2, 1), (1, 1), (5, 2), (2, 2), (1, 2), (2, 3), (1, 3), (2, 4), (1, 4), (2, 5), (1, 5), (6, 6), (2, 6), (1, 6), (6, 7), (2, 7), (6, 8), (2, 8), (6, 9), (2, 9), (6, 10), (2, 10), (6, 11), (2, 11), (6, 12), (2, 12), (6, 13), (2, 13), (2, 14), (2, 15), (6, 16), (2, 16), (6, 17), (2, 17), (6, 18), (2, 18), (6, 19), (2, 19), (6, 20), (2, 20), (6, 21), (2, 21), (6, 22), (2, 22), (6, 23), (2, 23), (6, 24), (2, 24), (6, 25), (2, 25), (6, 26), (2, 26), (6, 27), (2, 27), (2, 28), (2, 29), (2, 30), (2, 31), (2, 32), (2, 33), (2, 34), (2, 35), (2, 36), (2, 37), (2, 38), (2, 39), (2, 40), (2, 41), (2, 42), (2, 43), (2, 44), (11, 45), (2, 45), (11, 46), (2, 46), (11, 47), (2, 47), (11, 48), (2, 48), (11, 49), (2, 49), (11, 50), (2, 50), (2, 51), (13, 52), (2, 52), (2, 53), (2, 54), (2, 55), (2, 56), (2, 57), (2, 58), (2, 59), (2, 60), (2, 61), (2, 64), (14, 72), (12, 72), (14, 73), (14, 74), (15, 75), (14, 75), (15, 76), (14, 76), (15, 77), (14, 77), (15, 78), (15, 79), (15, 80), (15, 81), (15, 82), (18, 92), (18, 93), (18, 94), (18, 95), (18, 96), (13, 96), (21, 122), (21, 123), (21, 124), (21, 125), (21, 126), (21, 127), (21, 128), (21, 129), (21, 130), (21, 131), (21, 132), (21, 133), (24, 149), (24, 150), (24, 151)]}
    reid_col_all_cameras = []
    for cam in range(num_cameras):
        new_col_cam = np.ones((len(track_info[camera_names[cam]]),)) * -2
        for row_idx, row in enumerate(all_camera_matches):
            cam_tracks = all_camera_matches[row_idx][cam]
            for track in cam_tracks:
                idx_ = np.where(all_tracks[camera_names[cam]]["data"][:,1] == track)[0]
                new_col_cam[idx_] = row_idx+1
        reid_col_all_cameras.append(new_col_cam)

    reid = len(all_camera_matches)
    for cam in range(num_cameras):
        new_col = reid_col_all_cameras[cam]
        cam_data = all_tracks[camera_names[cam]]["data"]
        for i, new_id in enumerate(new_col):
            if new_id == -2:
                t = cam_data[:,1][i]
                idx = np.where(cam_data[:,1] == t)
                new_col[idx] = reid
                reid+=1
        
        # new_col = new_col.reshape(-1, 1)
        # new_cam_data = np.hstack((cam_data, new_col))
        new_cam_data = cam_data.copy()
        new_cam_data[:, 1] = new_col

        # Guardar los nuevos archivos con IDs actualizados

        np.savetxt(f"{camera_names[cam]}_renumbered_{threshold}_{window_size}.txt", new_cam_data, fmt="%d", delimiter=",")


    hotas = []
    idf1s = []
    for cam in range(num_cameras):
        cam_name = camera_names[cam]
        gt_path = f"/home/maria/mcv-c6-2025-team6/data/aic19-track1-mtmc-train/train/complete_videos/S01/gt/{cam_name}_gt.txt"
        det_path = f"{cam_name}_renumbered_{threshold}_{window_size}.txt"
        tracking = parse_tracking_file(gt_path)#
        # annotations = parse_tracking_file(f"{path_to_mtmc_train}/train/{seq}/{cam}/gt/gt.txt")
        annotations = parse_tracking_file(det_path)

        # hota, idf1 = get_hota_idf1(gt_path=gt_path, det_path=det_path, output_dir=os.path.join(os.path.dirname(det_path), f"metrics/{cam_name}"))
        result_hota, result_identity = calculate_metrics_taking_only_GTobject_into_account(tracking, annotations)
        hota = result_hota['HOTA']
        idf1 = result_identity['IDF1']
        hotas.append(np.mean(hota))
        idf1s.append(idf1)

    with open("experiments.txt", 'a') as exp:
        exp.write(f"Cameras: [{camera_names}], Threshold: {threshold}, Window size: {window_size}, HOTA (mean): {np.mean(hotas)}, IDF1: {np.mean(idf1s)}\n")

if __name__ == '__main__':
    # threshold = [random.uniform(0.85, 0.97) for _ in range(10)]
    threshold = [0.96]
    window_size = [0]
    # window_size = [5]
    for th in threshold:
        for ws in window_size:
            print(f"Experiment ws: {ws}, th: {th}")
            run_reid(ws, th)