import os
import json
import shutil

# ============================================================
# SCRIPT 3 V2 - ORGANIZAR DATASET CON 8 SUJETOS
# ============================================================


SESIONES = [
    {
        "sujeto": "GRABACION 1",
        "json":   r"D:\PROYECTO SEMINARIO VENEGAS NO BORRAR\dmd-dataset-distraction-gE-28\dmd\gE\28\s2\gE_28_s2_2019-03-15T10;12;30+01;00_rgb_ann_distraction.json",
    },
    {
        "sujeto": "GRABACION 2",
        "json":   r"D:\PROYECTO SEMINARIO VENEGAS NO BORRAR\dmd-dataset-distraction-gE-29\dmd\gE\29\s2\gE_29_s2_2019-03-15T13;42;24+01;00_rgb_ann_distraction.json",
    },
    {
        "sujeto": "GRABACION 3",
        "json":   r"D:\PROYECTO SEMINARIO VENEGAS NO BORRAR\dmd-dataset-distraction-gZ-36\dmd\gZ\36\s2\gZ_36_s2_2019-04-09T10;39;38+02;00_rgb_ann_distraction.json",
    },
    {
        "sujeto": "GRABACION 4",
        "json":   r"D:\PROYECTO SEMINARIO VENEGAS NO BORRAR\dmd-dataset-distraction-gZ-37\dmd\gZ\37\s2\gZ_37_s2_2019-04-08T15;45;15+02;00_rgb_ann_distraction.json",
    },
    {
        "sujeto": "GRABACION 5",
        "json":   r"D:\PROYECTO SEMINARIO VENEGAS NO BORRAR\dmd-dataset-rgb_ir-gB-10\dmd\gB\10\s2\gB_10_s2_2019-03-11T15;15;21+01;00_rgb_ann_distraction.json",
    },
    # 3 nuevos
    {
        "sujeto": "GRABACION 6",
        "json":   r"D:\PROYECTO SEMINARIO VENEGAS NO BORRAR\dmd-dataset-distraction-gB-6\dmd\gB\6\s2\gB_6_s2_2019-03-11T13;46;14+01;00_rgb_ann_distraction.json",
    },
    {
        "sujeto": "GRABACION 7",
        "json":   r"D:\PROYECTO SEMINARIO VENEGAS NO BORRAR\dmd-dataset-distraction-gB-7\dmd\gB\7\s2\gB_7_s2_2019-03-11T14;12;25+01;00_rgb_ann_distraction.json",
    },
    {
        "sujeto": "GRABACION 8",
        "json":   r"D:\PROYECTO SEMINARIO VENEGAS NO BORRAR\dmd-dataset-distraction-gB-9\dmd\gB\9\s2\gB_9_s2_2019-03-07T16;21;20+01;00_rgb_ann_distraction.json",
    },
]

frames_procesados_path = r"D:\PROYECTO SEMINARIO VENEGAS NO BORRAR\frames_procesados"
frames_aumentados_path = r"D:\PROYECTO SEMINARIO VENEGAS NO BORRAR\frames_aumentados"
output_path            = r"D:\PROYECTO SEMINARIO VENEGAS NO BORRAR\dataset_final_v2"

clases = {
    "driver_actions/safe_drive":      "safe_drive",
    "driver_actions/texting_right":   "texting_right",
    "driver_actions/texting_left":    "texting_left",
    "driver_actions/phonecall_right": "phonecall_right",
    "driver_actions/phonecall_left":  "phonecall_left",
    "driver_actions/reach_side":      "reach_side",
    "driver_actions/hair_and_makeup": "hair_and_makeup",
}

for carpeta in set(clases.values()):
    os.makedirs(os.path.join(output_path, carpeta), exist_ok=True)

print("Carpetas listas:")
for carpeta in sorted(set(clases.values())):
    print(f"  {output_path}\\{carpeta}")

contador_procesados_total = 0
contador_aumentados_total = 0

for sesion in SESIONES:

    sujeto    = sesion["sujeto"]
    json_path = sesion["json"]

    print(f"\n{'='*60}")
    print(f"  PROCESANDO SUJETO: {sujeto}")
    print(f"  JSON: {os.path.basename(json_path)}")
    print(f"{'='*60}")

    if not os.path.exists(json_path):
        print(f"  [ERROR] JSON no encontrado: {json_path}")
        continue

    with open(json_path, "r", encoding="utf-8") as archivo:
        data = json.load(archivo)

    acciones = data["openlabel"]["actions"]

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

                nombre_frame = f"{sujeto}_frame_{frame_real}.jpg"

                origen = os.path.join(frames_procesados_path, nombre_frame)

                if os.path.exists(origen):
                    nuevo_nombre = f"{sujeto}_{tipo.replace('/', '_')}_{frame_real}.jpg"
                    destino      = os.path.join(output_path, nombre_carpeta, nuevo_nombre)
                    if not os.path.exists(destino):
                        shutil.copy(origen, destino)
                        contador_procesados += 1

                for prefijo in ["flip", "bright", "rotated"]:
                    nombre_aug = f"{prefijo}_{sujeto}_frame_{frame_real}.jpg"
                    origen_aug = os.path.join(frames_aumentados_path, nombre_aug)
                    if os.path.exists(origen_aug):
                        nuevo_nombre_aug = f"{sujeto}_{tipo.replace('/', '_')}_{prefijo}_{frame_real}.jpg"
                        destino_aug      = os.path.join(output_path, nombre_carpeta, nuevo_nombre_aug)
                        if not os.path.exists(destino_aug):
                            shutil.copy(origen_aug, destino_aug)
                            contador_aumentados += 1

    print(f"  Procesados: {contador_procesados}  |  Aumentados: {contador_aumentados}")
    contador_procesados_total += contador_procesados
    contador_aumentados_total += contador_aumentados

print(f"\n{'='*60}")
print("  DATASET V2 ETIQUETADO GENERADO (8 sujetos)")
print(f"{'='*60}")
print(f"  Imágenes procesadas copiadas : {contador_procesados_total}")
print(f"  Imágenes aumentadas copiadas : {contador_aumentados_total}")
print(f"  Total imágenes copiadas      : {contador_procesados_total + contador_aumentados_total}")

print("\n  Distribución por clase:")
for carpeta in sorted(set(clases.values())):
    ruta  = os.path.join(output_path, carpeta)
    count = len([f for f in os.listdir(ruta) if f.endswith(".jpg")])
    print(f"    {carpeta:<25s}: {count:>5} imágenes")
print(f"{'='*60}")
