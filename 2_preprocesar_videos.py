import cv2
import os
import numpy as np

# ============================================================
# SCRIPT 2 - PREPROCESAMIENTO DEL VIDEO DE DISTRACCIONES
# ============================================================

# ============================================================
# RUTA DEL VIDEO
# ============================================================

video_path = r"C:\Users\PC\Desktop\CODIGO TESIS\dmd-dataset-rgb_ir-gB-10\dmd\gB\10\s2\gB_10_s2_2019-03-11T15;15;21+01;00_rgb_face.mp4"

# ============================================================
# CARPETAS DE SALIDA
# ============================================================

output_original  = "frames_originales"
output_processed = "frames_procesados"
output_augmented = "frames_aumentados"

os.makedirs(output_original,  exist_ok=True)
os.makedirs(output_processed, exist_ok=True)
os.makedirs(output_augmented, exist_ok=True)

# ============================================================
# FUNCIÓN PARA DETECTAR FRAMES BORROSOS
# ============================================================

def is_blurry(image, threshold=100):
    gray     = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold

# ============================================================
# ABRIR VIDEO
# ============================================================

cap = cv2.VideoCapture(video_path)

frame_count  = 0   # Número real del frame en el video
saved_count  = 0   # Cuántos frames se guardaron

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Extraer 1 frame cada 10
    if frame_count % 10 == 0:

        nombre = f"frame_{frame_count}.jpg"

        # Guardar original
        cv2.imwrite(os.path.join(output_original, nombre), frame)

        # Verificar blur
        if not is_blurry(frame):

            # ================================================
            # REDIMENSIONAMIENTO
            # ================================================

            resized = cv2.resize(frame, (224, 224))

            # ================================================
            # NORMALIZACIÓN
            # ================================================

            normalized = resized / 255.0
            processed  = (normalized * 255).astype("uint8")

            # Guardar procesado
            cv2.imwrite(os.path.join(output_processed, nombre), processed)

            # ================================================
            # DATA AUGMENTATION
            # ================================================

            # Flip horizontal
            flip = cv2.flip(processed, 1)

            # Cambio de brillo
            bright = cv2.convertScaleAbs(processed, alpha=1.2, beta=30)

            # Rotación leve
            h, w   = processed.shape[:2]
            matrix = cv2.getRotationMatrix2D((w // 2, h // 2), 10, 1.0)
            rotated = cv2.warpAffine(processed, matrix, (w, h))

            # Guardar augmentations con mismo número de frame
            cv2.imwrite(os.path.join(output_augmented, f"flip_{frame_count}.jpg"),    flip)
            cv2.imwrite(os.path.join(output_augmented, f"bright_{frame_count}.jpg"),  bright)
            cv2.imwrite(os.path.join(output_augmented, f"rotated_{frame_count}.jpg"), rotated)

            print(f"Frame procesado: {frame_count}")
            saved_count += 1

    frame_count += 1

cap.release()

print(f"\nPROCESAMIENTO FINALIZADO")
print(f"Frames totales del video : {frame_count}")
print(f"Frames guardados         : {saved_count}")
print(f"Imágenes aumentadas      : {saved_count * 3}")
