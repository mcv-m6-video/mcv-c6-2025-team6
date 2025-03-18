import os

def convert_yolo_to_output(input_dir, output_dir):
    # Asegurarse de que el directorio de salida exista
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Listar todos los archivos en el directorio de entrada
    filenames = os.listdir(input_dir)
    # Filtrar solo los archivos .txt
    txt_files = [f for f in filenames if f.endswith('.txt')]
    
    if len(txt_files) == 0:
        print("No se encontraron archivos .txt en el directorio de entrada.")
        return

    # Extraer el nombre del video a partir del primer archivo
    video_name = txt_files[0].split('_')[0]  # El video_name es la primera parte del nombre del archivo
    
    # Crear el archivo de salida en el directorio de salida con el nombre del video
    output_file_path = os.path.join(output_dir, f"{video_name}.txt")
    
    with open(output_file_path, 'w') as output_file:
        for filename in txt_files:
            # Extraer el frame_id del nombre del archivo (video_name_frame_id.txt)
            frame_id = filename.split('_')[-1].split('.')[0]
            
            # Abrir y procesar cada archivo de frame
            with open(os.path.join(input_dir, filename), 'r') as frame_file:
                for line in frame_file:
                    # Extraer la información del archivo YOLO
                    parts = line.strip().split()
                    class_id = parts[0]
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    score = float(parts[5])

                    # Convertir los valores normalizados de YOLO a coordenadas absolutas del cuadro delimitador
                    xtl = (x_center - width / 2)  # coordenada x superior izquierda
                    ytl = (y_center - height / 2)  # coordenada y superior izquierda
                    xbr = (x_center + width / 2)  # coordenada x inferior derecha
                    ybr = (y_center + height / 2)  # coordenada y inferior derecha

                    # Escribir la salida en el formato requerido
                    output_file.write(f"{frame_id} {class_id} {xtl} {ytl} {xbr} {ybr} {score}\n")

    print(f"Archivo de salida generado: {output_file_path}")

# Ejemplo de uso
input_directory = '/home/maria/mcv-c6-2025-team6/w4/output/runs/_predict1/labels'
output_directory = '/home/maria/mcv-c6-2025-team6/w4/output/runs/_predict1'

convert_yolo_to_output(input_directory, output_directory)
