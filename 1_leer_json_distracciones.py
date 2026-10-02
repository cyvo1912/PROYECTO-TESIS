import json
from pathlib import Path

from config_sesiones import sesiones_listas

SESIONES = sesiones_listas()

# ============================================================
# CLASES DENTRO DEL ALCANCE DE ESTE CICLO
# ============================================================

CLASES_CELULAR = {
    "driver_actions/phonecall_left",
    "driver_actions/phonecall_right",
    "driver_actions/texting_left",
    "driver_actions/texting_right",
}
CLASE_SEGURA = "driver_actions/safe_drive"
CLASES_RELEVANTES = CLASES_CELULAR | {CLASE_SEGURA}


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def frames_de_intervalos(intervalos):
    """Cuenta cuantos frames cubre una lista de frame_intervals."""
    total = 0
    for intervalo in intervalos:
        total += (intervalo["frame_end"] - intervalo["frame_start"] + 1)
    return total


def revisar_anotaciones_espaciales(data, mostrar_muestra=False):
    claves_openlabel = list(data["openlabel"].keys())
    objetos = data["openlabel"].get("objects", {})
    tiene_objetos = len(objetos) > 0

    if mostrar_muestra and tiene_objetos:
        print(f"\n  --- MUESTRA DE 'objects' ({len(objetos)} objeto(s) en total) ---")
        for obj_id, obj_info in list(objetos.items())[:5]:
            tipo_obj = obj_info.get("type", "(sin 'type')")
            claves_obj = list(obj_info.keys())
            print(f"    id={obj_id}  type={tipo_obj}  claves={claves_obj}")

            # Si trae 'object_data', ahi suele vivir la geometria (bbox, poly2d, etc.)
            object_data = obj_info.get("object_data", {})
            if object_data:
                print(f"      object_data -> claves: {list(object_data.keys())}")
                for geom_tipo, geom_lista in object_data.items():
                    if isinstance(geom_lista, list) and geom_lista:
                        primero = geom_lista[0]
                        print(f"        {geom_tipo}[0]: name={primero.get('name')}  val={primero.get('val')}")
        print("  --- FIN DE LA MUESTRA ---\n")

    return claves_openlabel, tiene_objetos


def inspeccionar_pointers_objetos(data):
    objetos = data["openlabel"].get("objects", {})
    print("\n  --- object_data_pointers por objeto ---")
    for obj_id, obj_info in objetos.items():
        tipo_obj = obj_info.get("type", "?")
        pointers = obj_info.get("object_data_pointers", {})
        print(f"    id={obj_id} ({tipo_obj}):")
        for nombre_ptr, info_ptr in pointers.items():
            print(f"      {nombre_ptr}: {info_ptr}")
    print("  --- FIN pointers ---\n")


def inspeccionar_frame_con_celular(data, tipo_accion="driver_actions/texting_left"):
    acciones = data["openlabel"]["actions"]
    frames_dict = data["openlabel"].get("frames", {})

    for accion_id, accion_info in acciones.items():
        if accion_info.get("type") != tipo_accion:
            continue
        intervalos = accion_info.get("frame_intervals", [])
        if not intervalos:
            continue

        frame_ejemplo = intervalos[0]["frame_start"]
        clave_frame = str(frame_ejemplo)

        print(f"  --- FRAME DE EJEMPLO ({tipo_accion}, frame {frame_ejemplo}) ---")
        if clave_frame not in frames_dict:
            print(f"    [!] El frame {frame_ejemplo} no aparece como clave directa en 'frames'.")
            print(f"        Claves de muestra disponibles en 'frames': {list(frames_dict.keys())[:5]}")
            return

        contenido_frame = frames_dict[clave_frame]
        print(f"    Claves del frame: {list(contenido_frame.keys())}")

        objetos_en_frame = contenido_frame.get("objects", {})
        if not objetos_en_frame:
            print("    Este frame no trae seccion 'objects' propia (probablemente no hay geometria por frame).")
        else:
            for obj_id, obj_frame_data in objetos_en_frame.items():
                print(f"    objeto {obj_id} en este frame: {obj_frame_data}")
        print("  --- FIN FRAME DE EJEMPLO ---\n")
        return

    print(f"  [!] No se encontro ninguna accion de tipo '{tipo_accion}' en esta sesion.\n")


# ============================================================
# LEER Y RESUMIR CADA SESION
# ============================================================

resumen_global = []

for sesion in SESIONES:

    sujeto = sesion["sujeto"]
    ruta_json = sesion["json"]

    print("\n" + "=" * 60)
    print(f"SUJETO: {sujeto}")
    print(f"JSON:   {Path(ruta_json).name}")
    print("=" * 60)

    if not Path(ruta_json).exists():
        print("  [ERROR] Archivo no encontrado, se omite esta sesion.")
        continue

    with open(ruta_json, "r", encoding="utf-8") as archivo:
        data = json.load(archivo)

    print("  JSON cargado correctamente")

    es_primera_sesion = (len(resumen_global) == 0)
    claves, tiene_objetos = revisar_anotaciones_espaciales(data, mostrar_muestra=es_primera_sesion)
    if es_primera_sesion and tiene_objetos:
        inspeccionar_pointers_objetos(data)
        inspeccionar_frame_con_celular(data, tipo_accion="driver_actions/texting_left")
    print(f"  Claves en 'openlabel'            : {claves}")
    print(f"  Incluye bounding boxes ('objects'): {'SI' if tiene_objetos else 'no'}")

    # --------------------------------------------------------
    # Acciones detectadas
    # --------------------------------------------------------
    acciones = data["openlabel"]["actions"]

    print("\n  ACCIONES DETECTADAS:")

    resumen_sesion = {"sujeto": sujeto, "clases": {}}

    for accion_id, accion_info in acciones.items():

        tipo = accion_info["type"]

        if "frame_intervals" not in accion_info:
            continue

        intervalos = accion_info["frame_intervals"]
        n_frames = frames_de_intervalos(intervalos)

        if tipo in CLASES_CELULAR:
            marca = "[CELULAR]"
        elif tipo == CLASE_SEGURA:
            marca = "[SEGURA]"
        else:
            marca = "[fuera de alcance]"

        print(f"    {marca:<18s} {tipo:<32s} frames: {n_frames:>5}  (intervalos: {len(intervalos)})")

        if tipo in CLASES_RELEVANTES:
            resumen_sesion["clases"][tipo] = resumen_sesion["clases"].get(tipo, 0) + n_frames

    resumen_global.append(resumen_sesion)
    print("\n  LECTURA FINALIZADA")

# ============================================================
# RESUMEN FINAL - SOLO CLASES DENTRO DEL ALCANCE (CELULAR)
# ============================================================

print("\n" + "=" * 60)
print("RESUMEN GLOBAL - FRAMES POR CLASE (ALCANCE: CELULAR)")
print("=" * 60)

totales = {}
for r in resumen_global:
    for clase, n in r["clases"].items():
        totales[clase] = totales.get(clase, 0) + n

for clase in sorted(CLASES_RELEVANTES):
    print(f"  {clase:<32s}: {totales.get(clase, 0):>6} frames")

print("\nPor sujeto:")
for r in resumen_global:
    if r["clases"]:
        detalle = ", ".join(f"{c.split('/')[-1]}={n}" for c, n in r["clases"].items())
    else:
        detalle = "(sin clases en alcance)"
    print(f"  {r['sujeto']:<14s}: {detalle}")
