import cv2
import os
import numpy as np

# ============================================================
# SCRIPT 2 - PREPROCESAMIENTO DE VIDEOS DE DISTRACCIONES
# ============================================================

SESIONES = [
    {
        "sujeto": "GRABACION 1",
        "video":  r"D:\CODIGO TESIS MEJORADO\dmd-dataset-distraction-gE-28\dmd\gE\28\s2\gE_28_s2_2019-03-15T10;12;30+01;00_rgb_face.mp4",
    },
    {
        "sujeto": "GRABACION 2",
        "video":  r"D:\CODIGO TESIS MEJORADO\dmd-dataset-distraction-gE-29\dmd\gE\29\s2\gE_29_s2_2019-03-15T13;42;24+01;00_rgb_face.mp4",
    },
    {
        "sujeto": "GRABACION 3",
        "video":  r"D:\CODIGO TESIS MEJORADO\dmd-dataset-distraction-gZ-36\dmd\gZ\36\s2\gZ_36_s2_2019-04-09T10;39;38+02;00_rgb_face.mp4",
    },
    {
        "sujeto": "GRABACION 4",
        "video":  r"D:\CODIGO TESIS MEJORADO\dmd-dataset-distraction-gZ-37\dmd\gZ\37\s2\gZ_37_s2_2019-04-08T15;45;15+02;00_rgb_face.mp4",
    },
    {
        "sujeto": "GRABACION 5",
        "video":  r"D:\CODIGO TESIS MEJORADO\dmd-dataset-rgb_ir-gB-10\dmd\gB\10\s2\gB_10_s2_2019-03-11T15;15;21+01;00_rgb_face.mp4",
    },
]

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
# PROCESAR CADA VIDEO
# ============================================================

total_guardados = 0

for sesion in SESIONES:

    sujeto     = sesion["sujeto"]
    video_path = sesion["video"]

    print(f"\n{'='*60}")
    print(f"  PROCESANDO SUJETO: {sujeto}")
    print(f"  Video: {os.path.basename(video_path)}")
    print(f"{'='*60}")

    if not os.path.exists(video_path):
        print(f"  [ERROR] Video no encontrado: {video_path}")
        continue

    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    saved_count = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # Extraer 1 frame cada 10
        if frame_count % 10 == 0:

            # Nombre con prefijo de sujeto para no pisarse
            nombre = f"{sujeto}_frame_{frame_count}.jpg"

            # Guardar original
            cv2.imwrite(os.path.join(output_original, nombre), frame)

            # Verificar blur
            if not is_blurry(frame):

                # ============================================
                # REDIMENSIONAMIENTO
                # ============================================

                resized = cv2.resize(frame, (224, 224))

                # ============================================
                # NORMALIZACIÓN
                # ============================================

                normalized = resized / 255.0
                processed  = (normalized * 255).astype("uint8")

                # Guardar procesado
                cv2.imwrite(os.path.join(output_processed, nombre), processed)

                # ============================================
                # DATA AUGMENTATION
                # ============================================

                # Flip horizontal
                flip = cv2.flip(processed, 1)

                # Cambio de brillo
                bright = cv2.convertScaleAbs(processed, alpha=1.2, beta=30)

                # Rotación leve
                h, w   = processed.shape[:2]
                matrix = cv2.getRotationMatrix2D((w // 2, h // 2), 10, 1.0)
                rotated = cv2.warpAffine(processed, matrix, (w, h))

                # Guardar augmentations con prefijo de sujeto
                cv2.imwrite(os.path.join(output_augmented, f"flip_{sujeto}_frame_{frame_count}.jpg"),    flip)
                cv2.imwrite(os.path.join(output_augmented, f"bright_{sujeto}_frame_{frame_count}.jpg"),  bright)
                cv2.imwrite(os.path.join(output_augmented, f"rotated_{sujeto}_frame_{frame_count}.jpg"), rotated)

                saved_count += 1

        frame_count += 1

    cap.release()

    print(f"  Frames totales del video : {frame_count}")
    print(f"  Frames guardados         : {saved_count}")
    print(f"  Imágenes aumentadas      : {saved_count * 3}")

    total_guardados += saved_count

print(f"\n{'='*60}")
print(f"  PREPROCESAMIENTO FINALIZADO")
print(f"  Total frames guardados (todos los sujetos): {total_guardados}")
print(f"  Total imágenes aumentadas                 : {total_guardados * 3}")
print(f"{'='*60}")
