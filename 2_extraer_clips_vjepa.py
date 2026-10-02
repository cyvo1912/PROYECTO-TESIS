import os
import json
import random

import cv2

from config_sesiones import sesiones_listas

# ============================================================
# SCRIPT 2 (SEMINARIO II) - EXTRACCION DE CLIPS PARA V-JEPA 2.1
# ------------------------------------------------------------

SESIONES = sesiones_listas()

CARPETA_CLASE = {
    "driver_actions/phonecall_left": "phonecall_left",
    "driver_actions/phonecall_right": "phonecall_right",
    "driver_actions/texting_left": "texting_left",
    "driver_actions/texting_right": "texting_right",
    "driver_actions/safe_drive": "safe_drive",
}
CLASES_RELEVANTES = set(CARPETA_CLASE.keys())

SUFIJO_VIDEO_ORIGEN = "_rgb_body.mp4"

# ------------------------------------------------------------
# PARAMETROS DEL CLIP - revisar antes de correr a gran escala
# ------------------------------------------------------------

LONGITUD_CLIP = 64                                      
                               
IMG_SIZE = 384                 
FPS_SALIDA = 30

MAX_CLIPS_POR_INTERVALO = 3    
                                
                                
MIN_FRACCION_REAL = 0.5        
                                
UMBRAL_BLUR = 100

APLICAR_FLIP_CON_SWAP_DE_ETIQUETA = False

SEMILLA = 42
random.seed(SEMILLA)

OUTPUT_DIR = "clips_vjepa"
MANIFIESTO_PATH = os.path.join(OUTPUT_DIR, "manifiesto_clips.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)
for carpeta in set(CARPETA_CLASE.values()):
    os.makedirs(os.path.join(OUTPUT_DIR, carpeta), exist_ok=True)


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def es_borroso(frame, umbral=UMBRAL_BLUR):
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gris, cv2.CV_64F).var() < umbral


def calidad_ventana(cap, inicio, fin):
    puntos = sorted(set([inicio, (inicio + fin) // 2, fin]))
    borrosos = 0
    for punto in puntos:
        cap.set(cv2.CAP_PROP_POS_FRAMES, punto)
        ok, frame = cap.read()
        if ok and es_borroso(frame):
            borrosos += 1
    return borrosos < len(puntos)


def generar_ventanas(inicio, fin):
    total = fin - inicio + 1

    if total < LONGITUD_CLIP * MIN_FRACCION_REAL:
        return []  # intervalo demasiado corto, ni con padding vale la pena

    if total < LONGITUD_CLIP:
        return [(inicio, fin, True)]  # una sola ventana, con padding

    n_disponibles = total // LONGITUD_CLIP
    n_ventanas = min(n_disponibles, MAX_CLIPS_POR_INTERVALO)

    if n_ventanas == n_disponibles:
        inicios = [inicio + i * LONGITUD_CLIP for i in range(n_ventanas)]
    else:
        paso = (total - LONGITUD_CLIP) / max(n_ventanas - 1, 1) if n_ventanas > 1 else 0
        inicios = [int(inicio + i * paso) for i in range(n_ventanas)]

    return [(ini, ini + LONGITUD_CLIP - 1, False) for ini in inicios]


def extraer_clip(cap, inicio, fin, requiere_padding):
    cap.set(cv2.CAP_PROP_POS_FRAMES, inicio)
    frames = []
    for _ in range(fin - inicio + 1):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(frame, (IMG_SIZE, IMG_SIZE)))

    if not frames:
        return None

    while requiere_padding and len(frames) < LONGITUD_CLIP:
        frames.append(frames[-1].copy())

    return frames


def guardar_clip_mp4(frames, ruta_salida, fps=FPS_SALIDA):
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(ruta_salida, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()


def aumentar_clip(frames):
    alpha = random.uniform(1.05, 1.25)
    beta = random.uniform(15, 35)
    angulo = random.uniform(-10, 10)

    h, w = frames[0].shape[:2]
    matriz = cv2.getRotationMatrix2D((w // 2, h // 2), angulo, 1.0)

    aumentados = []
    for f in frames:
        f2 = cv2.convertScaleAbs(f, alpha=alpha, beta=beta)
        f2 = cv2.warpAffine(f2, matriz, (w, h))
        aumentados.append(f2)
    return aumentados


# ============================================================
# EJECUCION PRINCIPAL
# ============================================================

filas_manifiesto = [
    "clip_id,sujeto,clase,ruta,frame_inicio,frame_fin,n_frames_reales,con_padding,aumentado"
]

contador_por_clase = {c: 0 for c in CARPETA_CLASE.values()}
descartados_blur = 0
descartados_corto = 0

for sesion in SESIONES:

    sujeto = sesion["sujeto"]
    ruta_json = sesion["json"]
    ruta_video = ruta_json.replace("_rgb_ann_distraction.json", SUFIJO_VIDEO_ORIGEN)

    print("=" * 60)
    print(f"SUJETO: {sujeto}")
    print("=" * 60)

    if not os.path.exists(ruta_json):
        print("  [ERROR] JSON no encontrado, se omite.")
        continue
    if not os.path.exists(ruta_video):
        print(f"  [ERROR] Video no encontrado: {ruta_video}")
        continue

    with open(ruta_json, "r", encoding="utf-8") as archivo:
        data = json.load(archivo)

    acciones = data["openlabel"]["actions"]
    cap = cv2.VideoCapture(ruta_video)

    clips_sesion = 0

    for accion_info in acciones.values():

        tipo = accion_info.get("type")
        if tipo not in CLASES_RELEVANTES:
            continue

        carpeta_clase = CARPETA_CLASE[tipo]
        intervalos = accion_info.get("frame_intervals", [])

        for intervalo in intervalos:

            inicio, fin = intervalo["frame_start"], intervalo["frame_end"]
            ventanas = generar_ventanas(inicio, fin)

            if not ventanas:
                descartados_corto += 1
                continue

            for (v_inicio, v_fin, requiere_padding) in ventanas:

                if not calidad_ventana(cap, v_inicio, v_fin):
                    descartados_blur += 1
                    continue

                frames = extraer_clip(cap, v_inicio, v_fin, requiere_padding)
                if frames is None:
                    continue

                clip_id = f"{sujeto}_{carpeta_clase}_{v_inicio}"
                ruta_clip = os.path.join(OUTPUT_DIR, carpeta_clase, f"{clip_id}.mp4")
                guardar_clip_mp4(frames, ruta_clip)

                contador_por_clase[carpeta_clase] += 1
                clips_sesion += 1
                filas_manifiesto.append(
                    f"{clip_id},{sujeto},{carpeta_clase},{ruta_clip},"
                    f"{v_inicio},{v_fin},{len(frames)},{requiere_padding},False"
                )

                # Variante aumentada: misma transformacion para todo el clip
                frames_aug = aumentar_clip(frames)
                clip_id_aug = f"{clip_id}_aug"
                ruta_clip_aug = os.path.join(OUTPUT_DIR, carpeta_clase, f"{clip_id_aug}.mp4")
                guardar_clip_mp4(frames_aug, ruta_clip_aug)

                contador_por_clase[carpeta_clase] += 1
                clips_sesion += 1
                filas_manifiesto.append(
                    f"{clip_id_aug},{sujeto},{carpeta_clase},{ruta_clip_aug},"
                    f"{v_inicio},{v_fin},{len(frames)},{requiere_padding},True"
                )

    cap.release()
    print(f"  Clips generados en esta sesion (con augmentation): {clips_sesion}")

# ------------------------------------------------------------
# MANIFIESTO Y RESUMEN
# ------------------------------------------------------------

with open(MANIFIESTO_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(filas_manifiesto) + "\n")

print("\n" + "=" * 60)
print("RESUMEN DE EXTRACCION DE CLIPS")
print("=" * 60)
for clase, n in sorted(contador_por_clase.items()):
    print(f"  {clase:<20s}: {n:>4} clips")

print(f"\n  Intervalos descartados por ser muy cortos : {descartados_corto}")
print(f"  Ventanas descartadas por blur              : {descartados_blur}")
print(f"\n  Manifiesto guardado en: {MANIFIESTO_PATH}")
print(f"  Clips guardados en    : {OUTPUT_DIR}/<clase>/")