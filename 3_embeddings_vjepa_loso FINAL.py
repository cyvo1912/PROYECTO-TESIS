import os
import csv

import numpy as np
import cv2
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "figure.dpi": 100,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "legend.fontsize": 8,
    "font.size": 10,
})

from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report, roc_curve, auc,
    accuracy_score, f1_score,
)

# ============================================================
# SCRIPT 3 (SEMINARIO II) - EMBEDDINGS DE V-JEPA 2.1 + FOLDS LOSO
# ============================================================

MANIFIESTO_CLIPS = os.path.join("clips_vjepa", "manifiesto_clips.csv")
CARPETA_EMBEDDINGS = "embeddings_vjepa"
RESULTADOS_LOSO = "resultados_loso_vjepa.csv"
CARPETA_REPORTES = "reportes_vjepa"
os.makedirs(CARPETA_EMBEDDINGS, exist_ok=True)
os.makedirs(CARPETA_REPORTES, exist_ok=True)

EPOCAS_PROBE = 200
LR_PROBE = 0.01
WEIGHT_DECAY_PROBE = 1e-3

NOMBRE_MODELO_HUB = "vjepa2_1_vit_base_384"   
IMG_SIZE = 384                                  
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


LIMITE_CLIPS_PRUEBA = None

BATCH_SIZE = 4

SEMILLA = 42
np.random.seed(SEMILLA)
torch.manual_seed(SEMILLA)


# ============================================================
# CARGA DEL MODELO (una sola vez)
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando device: {DEVICE}")
if DEVICE.type == "cpu":
    print("  [!] Sin GPU detectada: la extraccion de embeddings va a ser mas lenta.")
    print("      Sube LIMITE_CLIPS_PRUEBA (ej. 20) para una corrida corta primero.\n")

print(f"Cargando V-JEPA 2.1 ({NOMBRE_MODELO_HUB}) via torch.hub...")
try:
    resultado_carga = torch.hub.load("facebookresearch/vjepa2", NOMBRE_MODELO_HUB)
except Exception as error:
    print("  [ERROR] No se pudo cargar el modelo via torch.hub.")
    print(f"  Detalle: {error}")
    print("  Revisa: conexion a internet, 'pip install torch timm einops',")
    print("  y que la version de PyTorch sea razonablemente reciente.")
    raise

if isinstance(resultado_carga, (tuple, list)):
    modelo = resultado_carga[0]
    print(f"  [info] torch.hub devolvio {len(resultado_carga)} objetos; usando el primero (encoder).")
else:
    modelo = resultado_carga

modelo = modelo.to(DEVICE).eval()
print("  [OK] Modelo cargado.\n")


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def leer_manifiesto(ruta):
    with open(ruta, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def cargar_clip_como_tensor(ruta_clip, img_size=IMG_SIZE):
    cap = cv2.VideoCapture(ruta_clip)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if not frames:
        return None

    video = np.stack(frames, axis=0).astype(np.float32) / 255.0   # (T, H, W, C)
    video_t = torch.from_numpy(video).permute(3, 0, 1, 2)          # (C, T, H, W)

    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1, 1)
    video_t = (video_t - mean) / std

    return video_t  # (C, T, H, W)


def pooling_global(salida):
    if salida.ndim == 2:
        return salida
    salida_flat = salida.reshape(salida.shape[0], -1, salida.shape[-1])
    return salida_flat.mean(dim=1)


def extraer_embeddings_en_lote(rutas_clips, mostrar_forma=False):
    tensores = []
    indices_validos = []

    for i, ruta in enumerate(rutas_clips):
        t = cargar_clip_como_tensor(ruta)
        if t is not None:
            tensores.append(t)
            indices_validos.append(i)

    if not tensores:
        return {}

    lote = torch.stack(tensores, dim=0).to(DEVICE)  # (B, C, T, H, W)

    with torch.inference_mode():
        salida = modelo(lote)
        if mostrar_forma:
            print(f"  [debug] forma de salida del encoder (lote de {len(tensores)}): {tuple(salida.shape)}")
        embeddings = pooling_global(salida).cpu().numpy()  # (B, embed_dim)

    return {indices_validos[j]: embeddings[j] for j in range(len(indices_validos))}


# ============================================================
# PARTE A: EXTRAER (Y CACHEAR) EMBEDDINGS POR CLIP, EN LOTES
# ============================================================

filas = leer_manifiesto(MANIFIESTO_CLIPS)
if LIMITE_CLIPS_PRUEBA:
    filas = filas[:LIMITE_CLIPS_PRUEBA]
    print(f"[MODO PRUEBA] Procesando solo los primeros {LIMITE_CLIPS_PRUEBA} clips del manifiesto.\n")

print(f"Revisando cache de {len(filas)} clips...")

pendientes = [] 
cache_hits = 0

for fila in filas:
    clip_id = fila["clip_id"]
    ruta_npy = os.path.join(CARPETA_EMBEDDINGS, f"{clip_id}.npy")

    if os.path.exists(ruta_npy):
        cache_hits += 1
        continue

    ruta_clip = fila["ruta"]
    if not os.path.exists(ruta_clip):
        print(f"  [!] Clip no encontrado, se omite: {ruta_clip}")
        continue

    pendientes.append((clip_id, ruta_clip))

print(f"  {cache_hits} clips ya estaban en cache (se reutilizan, no se recalculan).")
print(f"  {len(pendientes)} clips por procesar, en lotes de {BATCH_SIZE}.\n")

primer_lote = True

for inicio in range(0, len(pendientes), BATCH_SIZE):

    lote = pendientes[inicio:inicio + BATCH_SIZE]
    rutas_lote = [ruta for (_, ruta) in lote]

    resultados = extraer_embeddings_en_lote(rutas_lote, mostrar_forma=primer_lote)
    primer_lote = False

    for j, (clip_id, ruta_clip) in enumerate(lote):
        if j not in resultados:
            print(f"  [!] No se pudo leer ningun frame de: {ruta_clip}")
            continue
        np.save(os.path.join(CARPETA_EMBEDDINGS, f"{clip_id}.npy"), resultados[j])

    procesados = min(inicio + BATCH_SIZE, len(pendientes))
    print(f"  {procesados}/{len(pendientes)} clips pendientes procesados")

# ------------------------------------------------------------
# Cargar TODOS los embeddings
# ------------------------------------------------------------

X, y, sujetos = [], [], []

for fila in filas:
    clip_id = fila["clip_id"]
    ruta_npy = os.path.join(CARPETA_EMBEDDINGS, f"{clip_id}.npy")
    if not os.path.exists(ruta_npy):
        continue
    X.append(np.load(ruta_npy))
    y.append(fila["clase"])
    sujetos.append(fila["sujeto"])

X = np.stack(X)
y = np.array(y)
sujetos = np.array(sujetos)

print(f"\n  Embeddings listos: X shape = {X.shape}")
print(f"  Guardados en      : {CARPETA_EMBEDDINGS}/ (se reutilizan en corridas futuras, no se recalculan)\n")


# ============================================================
# PARTE B: FOLDS LOSO CON PROBE LINEAL (PYTORCH) SOBRE LOS EMBEDDINGS
# ============================================================

codificador = LabelEncoder()
y_cod = codificador.fit_transform(y)
clases = codificador.classes_
n_clases = len(clases)


class ProbeLineal(nn.Module):
    def __init__(self, dim_entrada, n_clases):
        super().__init__()
        self.capa = nn.Linear(dim_entrada, n_clases)

    def forward(self, x):
        return self.capa(x)


def entrenar_fold(X_train, y_train, X_test, y_test):
    modelo_probe = ProbeLineal(X_train.shape[1], n_clases)
    optimizador = torch.optim.Adam(modelo_probe.parameters(), lr=LR_PROBE, weight_decay=WEIGHT_DECAY_PROBE)
    funcion_perdida = nn.CrossEntropyLoss()

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    perdidas_train, perdidas_test, accs_test = [], [], []

    for _ in range(EPOCAS_PROBE):
        modelo_probe.train()
        optimizador.zero_grad()
        perdida = funcion_perdida(modelo_probe(X_train_t), y_train_t)
        perdida.backward()
        optimizador.step()

        modelo_probe.eval()
        with torch.no_grad():
            logits_test_ep = modelo_probe(X_test_t)
            perdida_test = funcion_perdida(logits_test_ep, y_test_t)
            acc_test_ep = (logits_test_ep.argmax(dim=1) == y_test_t).float().mean().item()
        perdidas_train.append(perdida.item())
        perdidas_test.append(perdida_test.item())
        accs_test.append(acc_test_ep)

    modelo_probe.eval()
    with torch.no_grad():
        logits_test = modelo_probe(X_test_t)
        probas_test = torch.softmax(logits_test, dim=1).numpy()
        preds_test = logits_test.argmax(dim=1).numpy()

    return preds_test, probas_test, perdidas_train, perdidas_test, accs_test


sujetos_unicos = sorted(set(sujetos))
print("=" * 60)
print(f"EVALUACION LOSO ({len(sujetos_unicos)} folds, 1 por sujeto)")
print("=" * 60)

resultados_por_fold = []
curvas_por_fold = []
y_true_global, y_pred_global, probas_global = [], [], []

for sujeto_test in sujetos_unicos:

    idx_test = sujetos == sujeto_test
    idx_train = ~idx_test

    if len(set(y_cod[idx_train])) < 2 or idx_test.sum() == 0:
        print(f"  [!] {sujeto_test}: fold invalido (train o test insuficiente), se omite.")
        continue

    escalador = StandardScaler().fit(X[idx_train])
    X_train_s = escalador.transform(X[idx_train])
    X_test_s = escalador.transform(X[idx_test])

    preds, probas, p_train, p_test, acc_ep = entrenar_fold(
        X_train_s, y_cod[idx_train], X_test_s, y_cod[idx_test]
    )
    curvas_por_fold.append({
        "sujeto": sujeto_test,
        "train_loss": p_train, "test_loss": p_test, "test_acc": acc_ep,
    })

    acc = accuracy_score(y_cod[idx_test], preds)
    f1_macro = f1_score(y_cod[idx_test], preds, average="macro", zero_division=0)

    print(f"\n  Fold (test = {sujeto_test}) | n_test={idx_test.sum()}")
    print(f"    Accuracy : {acc:.4f}")
    print(f"    F1-macro : {f1_macro:.4f}")

    resultados_por_fold.append({
        "sujeto_test": sujeto_test,
        "n_test": int(idx_test.sum()),
        "accuracy": round(float(acc), 4),
        "f1_macro": round(float(f1_macro), 4),
    })

    y_true_global.extend(y_cod[idx_test])
    y_pred_global.extend(preds)
    probas_global.append(probas)

y_true_global = np.array(y_true_global)
y_pred_global = np.array(y_pred_global)
probas_global = np.concatenate(probas_global, axis=0)

# ------------------------------------------------------------
# RESUMEN AGREGADO
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("RESUMEN LOSO (agregado entre folds)")
print("=" * 60)

if resultados_por_fold:
    accs = [r["accuracy"] for r in resultados_por_fold]
    f1s = [r["f1_macro"] for r in resultados_por_fold]

    print(f"  Accuracy : {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
    print(f"  F1-macro : {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
    print("\n  Referencia Seminario I (YOLOv8 frame-a-frame): F1-macro = 0.6330")

    with open(RESULTADOS_LOSO, "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=["sujeto_test", "n_test", "accuracy", "f1_macro"])
        escritor.writeheader()
        escritor.writerows(resultados_por_fold)
    print(f"\n  Detalle por fold guardado en: {RESULTADOS_LOSO}")
else:
    print("  No se pudo completar ningun fold.")


# ============================================================
# REPORTES FINALES
# ============================================================

if resultados_por_fold:

    # --- 1) Reporte de clasificacion global ---
    reporte = classification_report(y_true_global, y_pred_global, target_names=clases,
                                     digits=3, zero_division=0)
    print("\n" + "=" * 60)
    print("REPORTE DE CLASIFICACION GLOBAL (LOSO, todos los folds combinados)")
    print("=" * 60)
    print(reporte)
    with open(os.path.join(CARPETA_REPORTES, "reporte_clasificacion.txt"), "w", encoding="utf-8") as f:
        f.write(reporte)

    # --- 2) Matriz de confusion global ---
    cm_abs = confusion_matrix(y_true_global, y_pred_global)
    cm_norm = confusion_matrix(y_true_global, y_pred_global, normalize="true")
 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
 
    disp1 = ConfusionMatrixDisplay(confusion_matrix=cm_abs, display_labels=clases)
    disp1.plot(ax=ax1, cmap="Blues", xticks_rotation=45, colorbar=True, values_format="d")
    ax1.set_title("Valores absolutos")
    ax1.grid(False)
 
    disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=clases)
    disp2.plot(ax=ax2, cmap="Blues", xticks_rotation=45, colorbar=True, values_format=".2f")
    ax2.set_title("Normalizada (por clase real)")
    ax2.set_ylabel("")
    ax2.grid(False)
 
    fig.suptitle("Matriz de Confusión — Solo V-JEPA (LOSO, todos los folds)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_REPORTES, "matriz_confusion.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # --- 4) Curvas de perdida + accuracy por fold 
    paleta20 = plt.get_cmap("tab20").colors
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(13, 5))
    for i, fold in enumerate(curvas_por_fold):
        color = paleta20[i % len(paleta20)]
        etiqueta = f"Fold {i} ({fold['sujeto']})"
        ax_loss.plot(fold["test_loss"], color=color, linewidth=1.3, label=etiqueta)
        ax_acc.plot(fold["test_acc"], color=color, linewidth=1.3, label=etiqueta)
    ax_loss.set_title("Val Loss por Fold")
    ax_loss.set_xlabel("Época")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(alpha=0.3)
    ax_acc.set_title("Val Accuracy por Fold")
    ax_acc.set_xlabel("Época")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.grid(alpha=0.3)
    ax_acc.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7, frameon=False)
    fig.suptitle("Comparación de Folds LOSO — Solo V-JEPA", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_REPORTES, "curvas_loss_accuracy.png"), dpi=200, bbox_inches="tight")
    plt.close()

    print(f"\nReportes guardados en: {CARPETA_REPORTES}/")
    print("  - reporte_clasificacion.txt")
    print("  - matriz_confusion.png")
    print("  - curvas_roc_auc.png")
    print("  - curvas_loss_accuracy.png")
