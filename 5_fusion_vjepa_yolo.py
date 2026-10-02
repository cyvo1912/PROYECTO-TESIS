import os
import csv

import numpy as np
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
# FUSION V-JEPA + YOLO (ABLATION STUDY)
# ============================================================

MANIFIESTO_CLIPS = os.path.join("clips_vjepa", "manifiesto_clips.csv")
CARPETA_EMBEDDINGS = "embeddings_vjepa"
EVIDENCIA_YOLO = os.path.join("clips_vjepa", "evidencia_yolo.csv")
CARPETA_REPORTES = "reportes_fusion"
os.makedirs(CARPETA_REPORTES, exist_ok=True)

EPOCAS = 200
LR = 0.01
WEIGHT_DECAY = 1e-3
SEMILLA = 42

torch.manual_seed(SEMILLA)
np.random.seed(SEMILLA)

CAMPOS_EVIDENCIA = [
    "celular_detectado", "frac_frames_con_celular", "conf_promedio",
    "conf_max", "cx_promedio", "cy_promedio", "area_promedio",
]


# ============================================================
# CARGAR TODO (embeddings + evidencia), YA CALCULADOS
# ============================================================

def leer_csv(ruta):
    with open(ruta, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


filas_manifiesto = leer_csv(MANIFIESTO_CLIPS)

if not os.path.exists(EVIDENCIA_YOLO):
    print(f"[ERROR] No se encontró {EVIDENCIA_YOLO}. Corre primero 4_evidencia_yolo_celular.py")
    raise SystemExit(1)

evidencia_por_clip = {
    fila["clip_id"]: [float(fila[campo]) for campo in CAMPOS_EVIDENCIA]
    for fila in leer_csv(EVIDENCIA_YOLO)
}

X_vjepa, X_yolo, y_texto, sujetos = [], [], [], []
faltantes_emb, faltantes_evid = 0, 0

for fila in filas_manifiesto:
    clip_id = fila["clip_id"]
    ruta_npy = os.path.join(CARPETA_EMBEDDINGS, f"{clip_id}.npy")

    if not os.path.exists(ruta_npy):
        faltantes_emb += 1
        continue
    if clip_id not in evidencia_por_clip:
        faltantes_evid += 1
        continue

    X_vjepa.append(np.load(ruta_npy))
    X_yolo.append(evidencia_por_clip[clip_id])
    y_texto.append(fila["clase"])
    sujetos.append(fila["sujeto"])

if faltantes_emb:
    print(f"[!] {faltantes_emb} clips sin embedding de V-JEPA, se omiten.")
if faltantes_evid:
    print(f"[!] {faltantes_evid} clips sin evidencia de YOLO, se omiten.")

X_vjepa = np.stack(X_vjepa)
X_yolo = np.stack(X_yolo)
X_fusion = np.concatenate([X_vjepa, X_yolo], axis=1)
sujetos = np.array(sujetos)

codificador = LabelEncoder()
y = codificador.fit_transform(y_texto)
clases = codificador.classes_
n_clases = len(clases)

print(f"Solo V-JEPA    : {X_vjepa.shape}")
print(f"Fusión (+YOLO) : {X_fusion.shape}  ({X_yolo.shape[1]} valores de evidencia agregados)")
print(f"Clases: {list(clases)}\n")


# ============================================================
# PROBE (idéntico al de reportes_vjepa_loso.py, para comparar limpio)
# ============================================================

class ProbeLineal(nn.Module):
    def __init__(self, dim_entrada, n_clases):
        super().__init__()
        self.capa = nn.Linear(dim_entrada, n_clases)

    def forward(self, x):
        return self.capa(x)


def entrenar_fold(X_train, y_train, X_test, y_test):
    modelo = ProbeLineal(X_train.shape[1], n_clases)
    optimizador = torch.optim.Adam(modelo.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    funcion_perdida = nn.CrossEntropyLoss()

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    perdidas_train, perdidas_test, accs_test = [], [], []

    for _ in range(EPOCAS):
        modelo.train()
        optimizador.zero_grad()
        perdida = funcion_perdida(modelo(X_train_t), y_train_t)
        perdida.backward()
        optimizador.step()

        modelo.eval()
        with torch.no_grad():
            logits_test_ep = modelo(X_test_t)
            perdida_test = funcion_perdida(logits_test_ep, y_test_t)
            acc_test_ep = (logits_test_ep.argmax(dim=1) == y_test_t).float().mean().item()
        perdidas_train.append(perdida.item())
        perdidas_test.append(perdida_test.item())
        accs_test.append(acc_test_ep)

    modelo.eval()
    with torch.no_grad():
        logits_test = modelo(X_test_t)
        probas_test = torch.softmax(logits_test, dim=1).numpy()
        preds_test = logits_test.argmax(dim=1).numpy()

    return preds_test, probas_test, perdidas_train, perdidas_test, accs_test


def evaluar_loso(X, y, sujetos, etiqueta):

    sujetos_unicos = sorted(set(sujetos))
    y_true_global, y_pred_global, probas_global = [], [], []
    resultados_fold = []
    curvas_por_fold = []  # una entrada por fold, sin promediar

    print(f"--- {etiqueta} ---")
    for sujeto_test in sujetos_unicos:
        idx_test = sujetos == sujeto_test
        idx_train = ~idx_test

        escalador = StandardScaler().fit(X[idx_train])
        X_train_s = escalador.transform(X[idx_train])
        X_test_s = escalador.transform(X[idx_test])

        preds, probas, p_train, p_test, acc_ep = entrenar_fold(
            X_train_s, y[idx_train], X_test_s, y[idx_test]
        )
        curvas_por_fold.append({
            "sujeto": sujeto_test,
            "train_loss": p_train, "test_loss": p_test, "test_acc": acc_ep,
        })

        acc = accuracy_score(y[idx_test], preds)
        f1_macro = f1_score(y[idx_test], preds, average="macro", zero_division=0)
        resultados_fold.append({"sujeto": sujeto_test, "accuracy": acc, "f1_macro": f1_macro})

        print(f"  {sujeto_test:<14s} | accuracy={acc:.4f} | f1_macro={f1_macro:.4f}")

        y_true_global.extend(y[idx_test])
        y_pred_global.extend(preds)
        probas_global.append(probas)

    y_true_global = np.array(y_true_global)
    y_pred_global = np.array(y_pred_global)
    probas_global = np.concatenate(probas_global, axis=0)

    accs = [r["accuracy"] for r in resultados_fold]
    f1s = [r["f1_macro"] for r in resultados_fold]
    print(f"  PROMEDIO: accuracy={np.mean(accs):.4f} +/- {np.std(accs):.4f}   "
          f"f1_macro={np.mean(f1s):.4f} +/- {np.std(f1s):.4f}\n")

    return {
        "resultados_fold": resultados_fold,
        "y_true": y_true_global, "y_pred": y_pred_global, "probas": probas_global,
        "accuracy_prom": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
        "f1_prom": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
        "curvas_por_fold": curvas_por_fold,
    }


# ============================================================
# CORRER LAS DOS VARIANTES
# ============================================================

resultado_vjepa = evaluar_loso(X_vjepa, y, sujetos, "SOLO V-JEPA (baseline)")
resultado_fusion = evaluar_loso(X_fusion, y, sujetos, "V-JEPA + YOLO (fusión)")


# ============================================================
# TABLA COMPARATIVA - el resultado clave del ablation study
# ============================================================

print("=" * 60)
print("COMPARACIÓN: SOLO V-JEPA  vs.  V-JEPA + YOLO")
print("=" * 60)
print(f"{'':<18s}{'Accuracy':>20s}{'F1-macro':>20s}")
print(f"{'Solo V-JEPA':<18s}{resultado_vjepa['accuracy_prom']:>12.4f} ± {resultado_vjepa['accuracy_std']:<5.3f}"
      f"{resultado_vjepa['f1_prom']:>12.4f} ± {resultado_vjepa['f1_std']:<5.3f}")
print(f"{'V-JEPA + YOLO':<18s}{resultado_fusion['accuracy_prom']:>12.4f} ± {resultado_fusion['accuracy_std']:<5.3f}"
      f"{resultado_fusion['f1_prom']:>12.4f} ± {resultado_fusion['f1_std']:<5.3f}")

diferencia_f1 = resultado_fusion["f1_prom"] - resultado_vjepa["f1_prom"]
print(f"\nDiferencia en F1-macro (fusión − solo V-JEPA): {diferencia_f1:+.4f}")
if diferencia_f1 > 0:
    print("  -> La evidencia de YOLO aportó, en promedio, sobre V-JEPA solo.")
elif diferencia_f1 < 0:
    print("  -> La fusión rindió peor que V-JEPA solo en este set de prueba.")
else:
    print("  -> Sin diferencia entre ambas variantes.")

with open(os.path.join(CARPETA_REPORTES, "comparacion_ablation.csv"), "w", newline="", encoding="utf-8") as f:
    escritor = csv.writer(f)
    escritor.writerow(["variante", "accuracy_promedio", "accuracy_std", "f1_macro_promedio", "f1_macro_std"])
    escritor.writerow(["solo_vjepa", resultado_vjepa["accuracy_prom"], resultado_vjepa["accuracy_std"],
                        resultado_vjepa["f1_prom"], resultado_vjepa["f1_std"]])
    escritor.writerow(["vjepa_yolo", resultado_fusion["accuracy_prom"], resultado_fusion["accuracy_std"],
                        resultado_fusion["f1_prom"], resultado_fusion["f1_std"]])


# ============================================================
# REPORTES GRAFICOS DE LA VARIANTE FUSIONADA (para el informe)
# ============================================================

def guardar_matriz_confusion(y_true, y_pred, titulo, ruta):
    cm_abs = confusion_matrix(y_true, y_pred)
    cm_norm = confusion_matrix(y_true, y_pred, normalize="true")
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
    fig.suptitle(titulo, fontweight="bold")
    plt.tight_layout()
    plt.savefig(ruta, dpi=200, bbox_inches="tight")
    plt.close()


def guardar_roc(y_true, probas, titulo, ruta):
    y_true_bin = label_binarize(y_true, classes=list(range(n_clases)))
    fig, ax = plt.subplots(figsize=(7.5, 6))
    paleta = plt.get_cmap("tab10").colors
    aucs = []
    for i, clase in enumerate(clases):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], probas[:, i])
        auc_clase = auc(fpr, tpr)
        aucs.append(auc_clase)
        ax.plot(fpr, tpr, color=paleta[i % len(paleta)], linewidth=2,
                label=f"{clase} (AUC={auc_clase:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#999999", linewidth=1, label="Azar (AUC=0.5)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title(f"{titulo}\nAUC promedio: {np.mean(aucs):.3f}")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(ruta, dpi=200, bbox_inches="tight")
    plt.close()


def graficar_curvas_por_fold(curvas_por_fold, titulo, ruta):
    """Reproduce el estilo de Seminario I: Loss y Accuracy por fold, una
    linea de color distinto por sujeto excluido, lado a lado."""
    paleta = plt.get_cmap("tab20").colors
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(13, 5))

    for i, fold in enumerate(curvas_por_fold):
        color = paleta[i % len(paleta)]
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

    fig.suptitle(titulo, fontweight="bold")
    plt.tight_layout()
    plt.savefig(ruta, dpi=200, bbox_inches="tight")
    plt.close()


guardar_matriz_confusion(
    resultado_fusion["y_true"], resultado_fusion["y_pred"],
    "Matriz de confusión — V-JEPA + YOLO (LOSO)",
    os.path.join(CARPETA_REPORTES, "matriz_confusion_fusion.png"),
)
guardar_roc(
    resultado_fusion["y_true"], resultado_fusion["probas"],
    "Curvas ROC — V-JEPA + YOLO (LOSO)",
    os.path.join(CARPETA_REPORTES, "curvas_roc_auc_fusion.png"),
)

# Curvas por fold (estilo Seminario I): una figura por variante
graficar_curvas_por_fold(
    resultado_vjepa["curvas_por_fold"],
    "Comparación de Folds LOSO — Solo V-JEPA",
    os.path.join(CARPETA_REPORTES, "curvas_loss_accuracy_solo_vjepa.png"),
)
graficar_curvas_por_fold(
    resultado_fusion["curvas_por_fold"],
    "Comparación de Folds LOSO — V-JEPA + YOLO",
    os.path.join(CARPETA_REPORTES, "curvas_loss_accuracy_fusion.png"),
)

reporte = classification_report(resultado_fusion["y_true"], resultado_fusion["y_pred"],
                                 target_names=clases, digits=3, zero_division=0)
with open(os.path.join(CARPETA_REPORTES, "reporte_clasificacion_fusion.txt"), "w", encoding="utf-8") as f:
    f.write(reporte)

print(f"\nReportes de la variante fusionada guardados en: {CARPETA_REPORTES}/")
print("(comparacion_ablation.csv es el resultado clave: solo V-JEPA vs V-JEPA+YOLO)")