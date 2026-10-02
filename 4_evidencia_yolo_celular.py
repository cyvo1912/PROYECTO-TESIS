import os
import csv

import cv2
from ultralytics import YOLO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8-whitegrid")

# ============================================================
# EXTRACCION DE EVIDENCIA YOLO (SOLO CELULAR) SOBRE LOS CLIPS
# ============================================================

MANIFIESTO_CLIPS = os.path.join("clips_vjepa", "manifiesto_clips.csv")
SALIDA_EVIDENCIA = os.path.join("clips_vjepa", "evidencia_yolo.csv")
CARPETA_REPORTES = "reportes_yolo"
os.makedirs(CARPETA_REPORTES, exist_ok=True)

FRAMES_MUESTREADOS_POR_CLIP = 5 
UMBRAL_CONFIANZA = 0.25

modelo = YOLO("yolov8n.pt")


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def leer_manifiesto(ruta):
    with open(ruta, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def muestrear_posiciones(total_frames, n_muestras):
    """Elige n_muestras posiciones repartidas a lo largo del clip."""
    if total_frames <= n_muestras:
        return list(range(total_frames))
    paso = total_frames / n_muestras
    return sorted(set(int(i * paso) for i in range(n_muestras)))


def evidencia_de_clip(ruta_clip):
    cap = cv2.VideoCapture(ruta_clip)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    posiciones = muestrear_posiciones(total, FRAMES_MUESTREADOS_POR_CLIP)

    detecciones = []
    for pos in posiciones:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ok, frame = cap.read()
        if not ok:
            continue

        resultado = modelo.predict(source=frame, conf=UMBRAL_CONFIANZA, verbose=False)[0]
        h, w = frame.shape[:2]

        for caja in resultado.boxes:
            if modelo.names[int(caja.cls[0])] != "cell phone":
                continue
            x1, y1, x2, y2 = map(float, caja.xyxy[0])
            conf = float(caja.conf[0])
            detecciones.append({
                "frame_relativo": pos,
                "conf": conf,
                "cx_norm": (x1 + x2) / 2 / w,
                "cy_norm": (y1 + y2) / 2 / h,
                "area_norm": ((x2 - x1) * (y2 - y1)) / (w * h),
            })

    cap.release()
    return detecciones, max(len(posiciones), 1)


def resumir_detecciones(detecciones, n_frames_muestreados):
    if not detecciones:
        return {
            "celular_detectado": 0,
            "frac_frames_con_celular": 0.0,
            "conf_promedio": 0.0,
            "conf_max": 0.0,
            "cx_promedio": -1.0,
            "cy_promedio": -1.0,
            "area_promedio": 0.0,
        }

    n_frames_con_deteccion = len(set(d["frame_relativo"] for d in detecciones))
    confs = [d["conf"] for d in detecciones]

    return {
        "celular_detectado": 1,
        "frac_frames_con_celular": round(n_frames_con_deteccion / n_frames_muestreados, 3),
        "conf_promedio": round(sum(confs) / len(confs), 3),
        "conf_max": round(max(confs), 3),
        "cx_promedio": round(sum(d["cx_norm"] for d in detecciones) / len(detecciones), 3),
        "cy_promedio": round(sum(d["cy_norm"] for d in detecciones) / len(detecciones), 3),
        "area_promedio": round(sum(d["area_norm"] for d in detecciones) / len(detecciones), 3),
    }


# ============================================================
# EJECUCION PRINCIPAL
# ============================================================

filas_manifiesto = leer_manifiesto(MANIFIESTO_CLIPS)
filas_salida = []

print(f"Procesando {len(filas_manifiesto)} clips del manifiesto...")

for i, fila in enumerate(filas_manifiesto, 1):

    ruta_clip = fila["ruta"]
    if not os.path.exists(ruta_clip):
        print(f"  [!] No encontrado, se omite: {ruta_clip}")
        continue

    detecciones, n_muestreados = evidencia_de_clip(ruta_clip)
    resumen = resumir_detecciones(detecciones, n_muestreados)

    filas_salida.append({
        "clip_id": fila["clip_id"],
        "sujeto": fila["sujeto"],
        "clase": fila["clase"],
        **resumen,
    })

    if i % 20 == 0 or i == len(filas_manifiesto):
        print(f"  {i}/{len(filas_manifiesto)} clips procesados")

# ------------------------------------------------------------
# GUARDAR CSV DE EVIDENCIA
# ------------------------------------------------------------

campos = ["clip_id", "sujeto", "clase", "celular_detectado", "frac_frames_con_celular",
          "conf_promedio", "conf_max", "cx_promedio", "cy_promedio", "area_promedio"]

with open(SALIDA_EVIDENCIA, "w", newline="", encoding="utf-8") as f:
    escritor = csv.DictWriter(f, fieldnames=campos)
    escritor.writeheader()
    escritor.writerows(filas_salida)

# ------------------------------------------------------------
# RESUMEN - clave para validar si la evidencia tiene sentido
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("RESUMEN DE EVIDENCIA YOLO")
print("=" * 60)

if filas_salida:
    n_con_celular = sum(1 for r in filas_salida if r["celular_detectado"] == 1)
    print(f"  Clips procesados               : {len(filas_salida)}")
    print(f"  Clips con al menos 1 deteccion : {n_con_celular} ({n_con_celular/len(filas_salida)*100:.1f}%)")

    print("\n  Por clase (esto es lo mas importante de revisar):")
    print("  -> phonecall_*/texting_* deberian salir con % alto de deteccion;")
    print("     safe_drive deberia salir bajo. Si sale al reves, el detector")
    print("     no esta funcionando bien en esta vista/camara.")

    clases_ordenadas = sorted(set(r["clase"] for r in filas_salida))
    resumen_por_clase = []

    for clase in clases_ordenadas:
        subset = [r for r in filas_salida if r["clase"] == clase]
        con_celular = sum(1 for r in subset if r["celular_detectado"] == 1)
        porcentaje = con_celular / len(subset) * 100
        print(f"    {clase:<20s}: {con_celular}/{len(subset)} clips con celular detectado "
              f"({porcentaje:.1f}%)")
        resumen_por_clase.append({
            "clase": clase, "n_clips": len(subset),
            "con_celular_detectado": con_celular, "porcentaje_deteccion": round(porcentaje, 1),
        })

    # --- Guardar resumen en CSV y texto plano ---
    ruta_resumen_csv = os.path.join(CARPETA_REPORTES, "resumen_deteccion_por_clase.csv")
    with open(ruta_resumen_csv, "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=["clase", "n_clips", "con_celular_detectado", "porcentaje_deteccion"])
        escritor.writeheader()
        escritor.writerows(resumen_por_clase)

    ruta_resumen_txt = os.path.join(CARPETA_REPORTES, "resumen_deteccion_por_clase.txt")
    with open(ruta_resumen_txt, "w", encoding="utf-8") as f:
        f.write(f"Clips procesados: {len(filas_salida)}\n")
        f.write(f"Clips con al menos 1 deteccion: {n_con_celular} ({n_con_celular/len(filas_salida)*100:.1f}%)\n\n")
        f.write("Por clase:\n")
        for r in resumen_por_clase:
            f.write(f"  {r['clase']:<20s}: {r['con_celular_detectado']}/{r['n_clips']} "
                    f"({r['porcentaje_deteccion']:.1f}%)\n")

    # --- Grafico de barras: % de deteccion por clase ---
    colores = ["#4C72B0" if "safe_drive" in c else "#DD8452" for c in clases_ordenadas]
    fig, ax = plt.subplots(figsize=(7, 5))
    barras = ax.bar(clases_ordenadas, [r["porcentaje_deteccion"] for r in resumen_por_clase], color=colores)
    ax.bar_label(barras, fmt="%.1f%%", padding=3)
    ax.set_ylabel("% de clips con celular detectado")
    ax.set_title("Tasa de detección de YOLOv8 (zero-shot) por clase")
    ax.set_ylim(0, max(r["porcentaje_deteccion"] for r in resumen_por_clase) * 1.25)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    ruta_grafico = os.path.join(CARPETA_REPORTES, "deteccion_por_clase.png")
    plt.savefig(ruta_grafico, dpi=200)
    plt.close()

    print(f"\n  Resumen guardado en: {ruta_resumen_csv}")
    print(f"  Resumen guardado en: {ruta_resumen_txt}")
    print(f"  Gráfico guardado en: {ruta_grafico}")
else:
    print("  No se proceso ningun clip. Revisa que MANIFIESTO_CLIPS apunte al archivo correcto.")

print(f"\n  Evidencia guardada en: {SALIDA_EVIDENCIA}")
