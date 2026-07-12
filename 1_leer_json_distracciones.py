import json

RUTAS_JSON = [
    r"D:\CODIGO TESIS MEJORADO\dmd-dataset-distraction-gE-28\dmd\gE\28\s2\gE_28_s2_2019-03-15T10;12;30+01;00_rgb_ann_distraction.json",
    r"D:\CODIGO TESIS MEJORADO\dmd-dataset-distraction-gE-29\dmd\gE\29\s2\gE_29_s2_2019-03-15T13;42;24+01;00_rgb_ann_distraction.json",
    r"D:\CODIGO TESIS MEJORADO\dmd-dataset-distraction-gZ-36\dmd\gZ\36\s2\gZ_36_s2_2019-04-09T10;39;38+02;00_rgb_ann_distraction.json",
    r"D:\CODIGO TESIS MEJORADO\dmd-dataset-distraction-gZ-37\dmd\gZ\37\s2\gZ_37_s2_2019-04-08T15;45;15+02;00_rgb_ann_distraction.json",
    r"D:\CODIGO TESIS MEJORADO\dmd-dataset-rgb_ir-gB-10\dmd\gB\10\s2\gB_10_s2_2019-03-11T15;15;21+01;00_rgb_ann_distraction.json",
]

# ============================================================
# LEER Y MOSTRAR ACCIONES
# ============================================================

for ruta_json in RUTAS_JSON:

    print("\n" + "="*60)
    print(f"JSON: {ruta_json.split(chr(92))[-1]}")
    print("="*60)

    with open(ruta_json, "r", encoding="utf-8") as archivo:
        data = json.load(archivo)

    print("\nJSON cargado correctamente\n")

    # ========================================================
    # ACCEDER A LAS ACCIONES
    # ========================================================

    acciones = data["openlabel"]["actions"]

    print("ACCIONES DETECTADAS EN EL DATASET:\n")

    # ========================================================
    # RECORRER ACCIONES
    # ========================================================

    for accion_id, accion_info in acciones.items():

        nombre_accion = accion_info["type"]

        print("===================================")
        print(f"ID ACCION: {accion_id}")
        print(f"TIPO: {nombre_accion}")

        # ====================================================
        # INTERVALOS DE FRAMES
        # ====================================================

        if "frame_intervals" in accion_info:

            intervalos = accion_info["frame_intervals"]

            for intervalo in intervalos:

                inicio = intervalo["frame_start"]
                fin    = intervalo["frame_end"]

                print(f"Frames: {inicio} -> {fin}")

    print("\nLECTURA FINALIZADA")
