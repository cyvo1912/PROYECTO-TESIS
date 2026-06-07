import os
import json
import shutil



json_path = r"C:\Users\PC\Desktop\CODIGO TESIS\dmd-dataset-rgb_ir-gB-10\dmd\gB\10\s2\gB_10_s2_2019-03-11T15;15;21+01;00_rgb_ann_distraction.json"

frames_procesados_path = r"C:\Users\PC\Desktop\PRUEBA CODIGO TESIS\frames_procesados"
frames_aumentados_path = r"C:\Users\PC\Desktop\PRUEBA CODIGO TESIS\frames_aumentados"

output_path = r"C:\Users\PC\Desktop\PRUEBA CODIGO TESIS\dataset_final"

# ============================================================
# MAPEO DE ETIQUETAS A CARPETAS (7 clases)
# ============================================================

clases = {
    "driver_actions/safe_drive":      "safe_drive",
    "driver_actions/texting_right":   "texting_right",
    "driver_actions/texting_left":    "texting_left",
    "driver_actions/phonecall_right": "phonecall_right",
    "driver_actions/phonecall_left":  "phonecall_left",
    "driver_actions/reach_side":      "reach_side",
    "driver_actions/hair_and_makeup": "hair_and_makeup",

}

# ============================================================
# CREAR CARPETAS DEL DATASET FINAL
# ============================================================

for carpeta in set(clases.values()):
    os.makedirs(os.path.join(output_path, carpeta), exist_ok=True)

print("Carpetas listas:")
for carpeta in sorted(set(clases.values())):
    print(f"  {output_path}\\{carpeta}")

# ============================================================
# LEER JSON
# ============================================================

with open(json_path, "r", encoding="utf-8") as archivo:
    data = json.load(archivo)

acciones = data["openlabel"]["actions"]

# ============================================================
# RECORRER ACCIONES Y COPIAR FRAMES
# ============================================================

contador_procesados = 0
contador_aumentados = 0

for accion_id in acciones:

    accion = acciones[accion_id]
    tipo   = accion["type"]

    if tipo not in clases:
        continue

    nombre_carpeta = clases[tipo]
    intervalos     = accion["frame_intervals"]

    for intervalo in intervalos:

        inicio = intervalo["frame_start"]
        fin    = intervalo["frame_end"]



        inicio_redondeado = (inicio // 10) * 10

        for frame_real in range(inicio_redondeado, fin + 1, 10):

            nombre_frame = f"frame_{frame_real}.jpg"

            # ================================================
            # COPIAR FRAME PROCESADO
            # ================================================

            origen = os.path.join(frames_procesados_path, nombre_frame)

            if os.path.exists(origen):

                nuevo_nombre = f"{tipo.replace('/', '_')}_{frame_real}.jpg"
                destino      = os.path.join(output_path, nombre_carpeta, nuevo_nombre)

                if not os.path.exists(destino):
                    shutil.copy(origen, destino)
                    contador_procesados += 1

            # ================================================
            # COPIAR FRAMES AUMENTADOS
            # ================================================

            for prefijo in ["flip", "bright", "rotated"]:

                nombre_aug = f"{prefijo}_{frame_real}.jpg"
                origen_aug = os.path.join(frames_aumentados_path, nombre_aug)

                if os.path.exists(origen_aug):

                    nuevo_nombre_aug = f"{tipo.replace('/', '_')}_{prefijo}_{frame_real}.jpg"
                    destino_aug      = os.path.join(output_path, nombre_carpeta, nuevo_nombre_aug)

                    if not os.path.exists(destino_aug):
                        shutil.copy(origen_aug, destino_aug)
                        contador_aumentados += 1

# ============================================================
# RESUMEN FINAL
# ============================================================

print("\nDATASET ETIQUETADO GENERADO")
print(f"Imágenes procesadas copiadas : {contador_procesados}")
print(f"Imágenes aumentadas copiadas : {contador_aumentados}")
print(f"Total imágenes copiadas      : {contador_procesados + contador_aumentados}")

print("\nDistribución por clase:")
for carpeta in sorted(set(clases.values())):
    ruta  = os.path.join(output_path, carpeta)
    count = len([f for f in os.listdir(ruta) if f.endswith(".jpg")])
    print(f"  {carpeta:<25s}: {count:>5} imágenes")