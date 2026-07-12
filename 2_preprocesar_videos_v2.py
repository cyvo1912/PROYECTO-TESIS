import cv2
import os
import numpy as np

# ============================================================
# SCRIPT 2 V2 - PREPROCESAMIENTO DE LOS 3 VIDEOS NUEVOS
# ============================================================

SESIONES = [
    {
        "sujeto": "GRABACION 6",
        "video":  r"D:\PROYECTO SEMINARIO VENEGAS NO BORRAR\dmd-dataset-distraction-gB-6\dmd\gB\6\s2\gB_6_s2_2019-03-11T13;46;14+01;00_rgb_face.mp4",
    },
    {
        "sujeto": "GRABACION 7",
        "video":  r"D:\PROYECTO SEMINARIO VENEGAS NO BORRAR\dmd-dataset-distraction-gB-7\dmd\gB\7\s2\gB_7_s2_2019-03-11T14;12;25+01;00_rgb_face.mp4",
    },
    {
        "sujeto": "GRABACION 8",
        "video":  r"D:\PROYECTO SEMINARIO VENEGAS NO BORRAR\dmd-dataset-distraction-gB-9\dmd\gB\9\s2\gB_9_s2_2019-03-07T16;21;20+01;00_rgb_face.mp4",
    },
]

output_original  = r"D:\PROYECTO SEMINARIO VENEGAS NO BORRAR\frames_originales"
output_processed = r"D:\PROYECTO SEMINARIO VENEGAS NO BORRAR\frames_procesados"
output_augmented = r"D:\PROYECTO SEMINARIO VENEGAS NO BORRAR\frames_aumentados"

os.makedirs(output_original,  exist_ok=True)
os.makedirs(output_processed, exist_ok=True)
os.makedirs(output_augmented, exist_ok=True)

def is_blurry(image, threshold=100):
    gray     = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold

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

        if frame_count % 10 == 0:
            nombre = f"{sujeto}_frame_{frame_count}.jpg"
            cv2.imwrite(os.path.join(output_original, nombre), frame)

            if not is_blurry(frame):
                resized    = cv2.resize(frame, (224, 224))
                normalized = resized / 255.0
                processed  = (normalized * 255).astype("uint8")

                cv2.imwrite(os.path.join(output_processed, nombre), processed)

                flip    = cv2.flip(processed, 1)
                bright  = cv2.convertScaleAbs(processed, alpha=1.2, beta=30)
                h, w    = processed.shape[:2]
                matrix  = cv2.getRotationMatrix2D((w // 2, h // 2), 10, 1.0)
                rotated = cv2.warpAffine(processed, matrix, (w, h))

                cv2.imwrite(os.path.join(output_augmented, f"flip_{sujeto}_frame_{frame_count}.jpg"),    flip)
                cv2.imwrite(os.path.join(output_augmented, f"bright_{sujeto}_frame_{frame_count}.jpg"),  bright)
                cv2.imwrite(os.path.join(output_augmented, f"rotated_{sujeto}_frame_{frame_count}.jpg"), rotated)

                saved_count += 1
        frame_count += 1

    cap.release()
    print(f"  Frames totales: {frame_count} | Guardados: {saved_count} | Aumentados: {saved_count * 3}")
    total_guardados += saved_count

print(f"\n{'='*60}")
print(f"  PREPROCESAMIENTO V2 FINALIZADO")
print(f"  Total frames nuevos: {total_guardados}")
print(f"  Total aumentados:    {total_guardados * 3}")
print(f"{'='*60}")
