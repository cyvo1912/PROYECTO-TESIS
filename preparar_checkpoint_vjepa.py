import os
import urllib.request

import torch


NOMBRE_ARCHIVO = "vjepa2_1_vitb_dist_vitG_384.pt"
URL_CORRECTA = f"https://dl.fbaipublicfiles.com/vjepa2/{NOMBRE_ARCHIVO}"

carpeta_checkpoints = os.path.join(torch.hub.get_dir(), "checkpoints")
os.makedirs(carpeta_checkpoints, exist_ok=True)

ruta_destino = os.path.join(carpeta_checkpoints, NOMBRE_ARCHIVO)

print(f"Carpeta de checkpoints detectada: {carpeta_checkpoints}")

if os.path.exists(ruta_destino):
    tam_mb = os.path.getsize(ruta_destino) / (1024 * 1024)
    print(f"[OK] El checkpoint ya existe ({tam_mb:.1f} MB): {ruta_destino}")
    print("     No hace falta descargar de nuevo.")
else:
    print("Descargando checkpoint (puede tardar varios minutos segun tu conexion)...")
    print(f"  Desde: {URL_CORRECTA}")
    print(f"  Hacia: {ruta_destino}")

    def mostrar_progreso(bloque_num, tam_bloque, tam_total):
        descargado = bloque_num * tam_bloque
        if tam_total > 0:
            porcentaje = min(100, descargado * 100 // tam_total)
            print(f"\r  {porcentaje:>3d}%  ({descargado // (1024*1024)} MB / {tam_total // (1024*1024)} MB)",
                  end="", flush=True)

    try:
        urllib.request.urlretrieve(URL_CORRECTA, ruta_destino, reporthook=mostrar_progreso)
        print("\n[OK] Descarga completa.")
    except Exception as error:
        print(f"\n[ERROR] Fallo la descarga: {error}")
        if os.path.exists(ruta_destino):
            os.remove(ruta_destino)  # evita dejar un archivo a medio descargar con el nombre correcto
        raise

print("\nListo. Ahora corre 3_embeddings_vjepa_loso.py normalmente.")
