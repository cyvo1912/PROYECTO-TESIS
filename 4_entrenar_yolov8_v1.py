import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

# ============================================================
# SCRIPT 4 V1 - ENTRENAMIENTO YOLOV8 CON LOSO (5 sujetos)
# ============================================================


from ultralytics import YOLO
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ============================================================
# CONFIGURACIÓN
# ============================================================

DATASET_RAIZ  = r"D:\PROYECTO SEMINARIO VENEGAS NO BORRAR\dataset_final"
DATASET_LOSO  = r"D:\PROYECTO SEMINARIO VENEGAS NO BORRAR\dataset_loso_v1"

MODELO_BASE   = "yolov8n-cls.pt"
EPOCHS        = 50
IMG_SIZE      = 224
BATCH_SIZE    = 64
DEVICE        = 0                    
LR0           = 0.01
DROPOUT       = 0.0
OPTIMIZER     = 'Adam'

PROYECTO      = "resultados_v1_loso"
NOMBRE_BASE   = "yolov8n_loso"

CLASES = [
    "hair_and_makeup",
    "phonecall_left",
    "phonecall_right",
    "reach_side",
    "safe_drive",
    "texting_left",
    "texting_right",
]

# ============================================================
# SUJETOS
# ============================================================

SUJETO_TEST = "GRABACION 2"                                         
SUJETOS_CV  = ["GRABACION 1", "GRABACION 3", "GRABACION 4", "GRABACION 5"]  
N_FOLDS     = len(SUJETOS_CV)                                       

SEMILLA = 42

RESULTADOS_DIR = os.path.join(PROYECTO, NOMBRE_BASE, "resultados_loso")
os.makedirs(RESULTADOS_DIR, exist_ok=True)

colores_fold = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]


# ============================================================
# FUNCIÓN: obtener imágenes de un sujeto en una clase
# ============================================================

def obtener_imagenes_sujeto(clase, sujeto):
    """Devuelve las imágenes de un sujeto específico en una clase."""
    ruta_clase = os.path.join(DATASET_RAIZ, clase)
    return [
        img for img in Path(ruta_clase).glob("*.jpg")
        if img.name.startswith(f"{sujeto}_")
    ]


# ============================================================
# PASO 1 - PREPARAR CARPETAS PARA UN FOLD
# ============================================================

def preparar_fold(fold_idx):

    if os.path.exists(DATASET_LOSO):
        shutil.rmtree(DATASET_LOSO)

    for split in ["train", "val"]:
        for clase in CLASES:
            os.makedirs(os.path.join(DATASET_LOSO, split, clase), exist_ok=True)

    sujeto_val    = SUJETOS_CV[fold_idx]
    sujetos_train = [s for s in SUJETOS_CV if s != sujeto_val]

    print(f"    Train: {sujetos_train}")
    print(f"    Val:   {sujeto_val}")

    for clase in CLASES:

        for sujeto in sujetos_train:
            imagenes = obtener_imagenes_sujeto(clase, sujeto)
            for img in imagenes:
                shutil.copy(str(img), os.path.join(DATASET_LOSO, "train", clase, img.name))

        # Val = 1 sujeto
        imagenes_val = obtener_imagenes_sujeto(clase, sujeto_val)
        for img in imagenes_val:
            shutil.copy(str(img), os.path.join(DATASET_LOSO, "val", clase, img.name))


# ============================================================
# PASO 2 - PREPARAR TEST SET
# ============================================================

def preparar_test():

    print(f"\n  Preparando Test set con sujeto: {SUJETO_TEST}")

    for clase in CLASES:
        os.makedirs(os.path.join(DATASET_LOSO, "test", clase), exist_ok=True)
        imagenes = obtener_imagenes_sujeto(clase, SUJETO_TEST)
        for img in imagenes:
            shutil.copy(str(img), os.path.join(DATASET_LOSO, "test", clase, img.name))
        print(f"    {clase:<25s}: {len(imagenes):>4} imágenes")


# ============================================================
# PASO 3 - ENTRENAR UN FOLD
# ============================================================

def entrenar_fold(fold_num):

    nombre_run = f"{NOMBRE_BASE}_fold{fold_num}"

    print(f"\n  Entrenando Fold {fold_num} ({EPOCHS} épocas)...")

    model = YOLO(MODELO_BASE)
    model.train(
        data      = DATASET_LOSO,
        epochs    = EPOCHS,
        imgsz     = IMG_SIZE,
        batch     = BATCH_SIZE,
        device    = DEVICE,
        project   = PROYECTO,
        name      = nombre_run,
        exist_ok  = True,
        verbose   = False,
        lr0       = LR0,
        dropout   = DROPOUT,
        optimizer = OPTIMIZER,
    )

    mejor = os.path.join("runs", "classify", PROYECTO, nombre_run, "weights", "best.pt")
    if not os.path.exists(mejor):
        mejor = os.path.join(PROYECTO, nombre_run, "weights", "best.pt")

    return YOLO(mejor)


# ============================================================
# PASO 4 - EVALUAR SOBRE UN CONJUNTO
# ============================================================

def evaluar_sobre(model, split="val"):

    y_true = []
    y_pred = []

    for idx_clase, clase in enumerate(CLASES):
        ruta_clase = os.path.join(DATASET_LOSO, split, clase)
        if not os.path.exists(ruta_clase):
            continue
        imagenes = list(Path(ruta_clase).glob("*.jpg"))

        for img_path in imagenes:
            resultado = model.predict(source=str(img_path), verbose=False)
            pred_idx  = int(resultado[0].probs.top1)
            y_true.append(idx_clase)
            y_pred.append(pred_idx)

    return np.array(y_true), np.array(y_pred)


# ============================================================
# EJECUCIÓN PRINCIPAL
# ============================================================

if __name__ == "__main__":

    print("\n" + "="*60)
    print("  YOLOV8 V1 - LEAVE-ONE-SUBJECT-OUT (LOSO)")
    print(f"  Sujeto Test: {SUJETO_TEST}")
    print(f"  Sujetos CV:  {SUJETOS_CV} ({N_FOLDS} folds)")
    print(f"  {MODELO_BASE} | {EPOCHS} épocas | {OPTIMIZER}")
    print("="*60)

    # ========================================================
    # CROSS-VALIDATION POR SUJETO
    # ========================================================

    resultados_cv = []

    for fold in range(N_FOLDS):

        print("\n" + "="*60)
        print(f"  FOLD {fold} / {N_FOLDS - 1}  (Val = {SUJETOS_CV[fold]})")
        print("="*60)

        preparar_fold(fold)
        model = entrenar_fold(fold)

        print(f"  Evaluando Fold {fold} sobre Val ({SUJETOS_CV[fold]})...")
        y_true, y_pred = evaluar_sobre(model, "val")

        prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        rec  = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1   = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        acc  = accuracy_score(y_true, y_pred)

        resultados_cv.append({
            "fold": fold, "sujeto_val": SUJETOS_CV[fold],
            "precision": prec, "recall": rec, "f1": f1, "accuracy": acc,
        })

        print(f"\n  Fold {fold} ({SUJETOS_CV[fold]}) -> P: {prec:.4f} | R: {rec:.4f} | F1: {f1:.4f} | Acc: {acc:.4f}")

    # ========================================================
    # TABLA RESUMEN CV
    # ========================================================

    print("\n" + "="*60)
    print("  RESUMEN CROSS-VALIDATION (por sujeto)")
    print("="*60)
    print(f"\n  {'':<22s}{'Precision':>12s}{'Recall':>12s}{'F1':>12s}{'Accuracy':>12s}")
    print("  " + "-"*66)

    for r in resultados_cv:
        print(f"  Fold {r['fold']} ({r['sujeto_val']:<12s}){r['precision']:>12.4f}{r['recall']:>12.4f}{r['f1']:>12.4f}{r['accuracy']:>12.4f}")

    prom_prec = np.mean([r["precision"] for r in resultados_cv])
    prom_rec  = np.mean([r["recall"]    for r in resultados_cv])
    prom_f1   = np.mean([r["f1"]        for r in resultados_cv])
    prom_acc  = np.mean([r["accuracy"]  for r in resultados_cv])
    std_f1    = np.std([r["f1"]         for r in resultados_cv])

    print("  " + "-"*66)
    print(f"  {'Promedio':<22s}{prom_prec:>12.4f}{prom_rec:>12.4f}{prom_f1:>12.4f}{prom_acc:>12.4f}")
    print(f"\n  Desviación estándar F1: {std_f1:.4f}")

    # ========================================================
    # EVALUACIÓN FINAL SOBRE SUJETO TEST
    # ========================================================

    print("\n" + "="*60)
    print(f"  EVALUACIÓN FINAL SOBRE SUJETO TEST: {SUJETO_TEST}")
    print("="*60)

    mejor_fold = max(resultados_cv, key=lambda x: x["f1"])
    print(f"  Mejor fold: {mejor_fold['fold']} ({mejor_fold['sujeto_val']}) con F1={mejor_fold['f1']:.4f}")

    nombre_run = f"{NOMBRE_BASE}_fold{mejor_fold['fold']}"
    mejor_modelo = os.path.join("runs", "classify", PROYECTO, nombre_run, "weights", "best.pt")
    if not os.path.exists(mejor_modelo):
        mejor_modelo = os.path.join(PROYECTO, nombre_run, "weights", "best.pt")

    model = YOLO(mejor_modelo)

    preparar_test()

    y_true_test, y_pred_test = evaluar_sobre(model, "test")

    reporte = classification_report(
        y_true_test, y_pred_test,
        target_names=CLASES, digits=4, zero_division=0
    )
    acc_test = accuracy_score(y_true_test, y_pred_test)
    f1_test  = f1_score(y_true_test, y_pred_test, average="weighted", zero_division=0)

    print("\n" + reporte)
    print(f"  Accuracy Test : {acc_test:.4f}")
    print(f"  F1 Test       : {f1_test:.4f}")

    # ========================================================
    # GUARDAR REPORTE TXT
    # ========================================================

    ruta_txt = os.path.join(RESULTADOS_DIR, "reporte_loso_v1.txt")
    with open(ruta_txt, "w", encoding="utf-8") as f:
        f.write("REPORTE LOSO V1 - YOLOv8n Distracciones\n")
        f.write("="*60 + "\n")
        f.write(f"Modelo: {MODELO_BASE} | Épocas: {EPOCHS} | LR: {LR0}\n")
        f.write(f"Dropout: {DROPOUT} | Optimizer: {OPTIMIZER}\n")
        f.write(f"Sujeto Test: {SUJETO_TEST} | Sujetos CV: {SUJETOS_CV}\n")
        f.write("="*60 + "\n\n")

        f.write("CROSS-VALIDATION POR SUJETO:\n")
        f.write(f"{'':<22s}{'Precision':>12s}{'Recall':>12s}{'F1':>12s}{'Accuracy':>12s}\n")
        f.write("-"*66 + "\n")
        for r in resultados_cv:
            f.write(f"Fold {r['fold']} ({r['sujeto_val']:<12s}){r['precision']:>12.4f}{r['recall']:>12.4f}{r['f1']:>12.4f}{r['accuracy']:>12.4f}\n")
        f.write("-"*66 + "\n")
        f.write(f"{'Promedio':<22s}{prom_prec:>12.4f}{prom_rec:>12.4f}{prom_f1:>12.4f}{prom_acc:>12.4f}\n")
        f.write(f"\nDesviación estándar F1: {std_f1:.4f}\n\n")

        f.write(f"\nEVALUACIÓN FINAL SOBRE SUJETO TEST ({SUJETO_TEST}):\n")
        f.write("="*60 + "\n")
        f.write(reporte)
        f.write(f"\nAccuracy Test : {acc_test:.4f}\n")
        f.write(f"F1 Test       : {f1_test:.4f}\n")

    print(f"\n  [OK] Reporte guardado en: {ruta_txt}")

    # ========================================================
    # GRÁFICO DE BARRAS POR FOLD
    # ========================================================

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(N_FOLDS)
    ancho = 0.25

    ax.bar(x - ancho, [r["precision"] for r in resultados_cv], ancho, label="Precision", color="#2196F3")
    ax.bar(x,         [r["recall"]    for r in resultados_cv], ancho, label="Recall",    color="#4CAF50")
    ax.bar(x + ancho, [r["f1"]        for r in resultados_cv], ancho, label="F1",        color="#FF9800")

    ax.set_title("Métricas por Fold - LOSO V1", fontsize=13, fontweight="bold")
    ax.set_xlabel("Fold (Sujeto de Validación)")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {i}\n({SUJETOS_CV[i]})" for i in range(N_FOLDS)])
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    ruta_graf = os.path.join(RESULTADOS_DIR, "metricas_por_fold_loso_v1.png")
    plt.savefig(ruta_graf, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Gráfico guardado en: {ruta_graf}")

    # ========================================================
    # MATRIZ DE CONFUSIÓN TEST
    # ========================================================

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(
        f"Matriz de Confusión - Sujeto Test: {SUJETO_TEST}",
        fontsize=14, fontweight="bold"
    )

    for ax, normalize, titulo, fmt in zip(
        axes, [None, "true"],
        ["Valores Absolutos", "Normalizada (%)"], ["d", ".2f"]
    ):
        cm = confusion_matrix(y_true_test, y_pred_test, normalize=normalize)
        sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=CLASES, yticklabels=CLASES,
                    ax=ax, linewidths=0.5)
        ax.set_title(titulo, fontsize=12)
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Real")
        ax.tick_params(axis="x", rotation=35)
        ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    ruta_cm = os.path.join(RESULTADOS_DIR, "matriz_confusion_test_v1.png")
    plt.savefig(ruta_cm, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Matriz de confusión guardada en: {ruta_cm}")

    # ========================================================
    # CURVAS DE PÉRDIDA POR FOLD
    # ========================================================

    csvs = []
    for fold in range(N_FOLDS):
        nombre_run = f"{NOMBRE_BASE}_fold{fold}"
        ruta_csv = os.path.join("runs", "classify", PROYECTO, nombre_run, "results.csv")
        if not os.path.exists(ruta_csv):
            ruta_csv = os.path.join(PROYECTO, nombre_run, "results.csv")
        if os.path.exists(ruta_csv):
            df = pd.read_csv(ruta_csv)
            df.columns = [c.strip() for c in df.columns]
            csvs.append((fold, df))

    if csvs:

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Comparación de Folds LOSO V1 - Loss y Accuracy", fontsize=14, fontweight="bold")

        for i, (fold, df) in enumerate(csvs):
            col_val_loss = [c for c in df.columns if "val" in c and "loss" in c]
            col_acc      = [c for c in df.columns if "acc" in c.lower() and "top1" in c.lower()]
            epochs_x = df["epoch"] if "epoch" in df.columns else range(1, len(df) + 1)

            for col in col_val_loss:
                axes[0].plot(epochs_x, df[col], label=f"Fold {fold} ({SUJETOS_CV[fold]})",
                            color=colores_fold[i], linewidth=1.5)
            for col in col_acc:
                axes[1].plot(epochs_x, df[col], label=f"Fold {fold} ({SUJETOS_CV[fold]})",
                            color=colores_fold[i], linewidth=1.5)

        axes[0].set_title("Val Loss por Fold", fontsize=12)
        axes[0].set_xlabel("Época")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].set_title("Val Accuracy por Fold", fontsize=12)
        axes[1].set_xlabel("Época")
        axes[1].set_ylabel("Accuracy")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        ruta_comp = os.path.join(RESULTADOS_DIR, "comparacion_folds_loso_v1.png")
        plt.savefig(ruta_comp, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [OK] Comparación de folds guardada en: {ruta_comp}")

    # ========================================================
    # F1 POR CLASE (TEST)
    # ========================================================

    f1_scores = f1_score(y_true_test, y_pred_test, average=None, zero_division=0)
    colores_clases = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0", "#F44336", "#00BCD4", "#FF5722"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(CLASES, f1_scores, color=colores_clases, edgecolor="white", linewidth=0.8)
    ax.set_title(f"F1-Score por Clase - Test ({SUJETO_TEST})", fontsize=13, fontweight="bold")
    ax.set_ylabel("F1-Score")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(y=np.mean(f1_scores), color="red", linestyle="--", alpha=0.7,
               label=f"Promedio: {np.mean(f1_scores):.3f}")
    ax.legend()
    for bar, val in zip(bars, f1_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    ruta_f1 = os.path.join(RESULTADOS_DIR, "f1_por_clase_test_v1.png")
    plt.savefig(ruta_f1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] F1 por clase guardado en: {ruta_f1}")

    print(f"\n{'='*60}")
    print(f"  LOSO V1 COMPLETADO")
    print(f"  Resultados en: {RESULTADOS_DIR}")
    print(f"{'='*60}")
    print(f"""
  ARCHIVOS GENERADOS:
    {RESULTADOS_DIR}/
      reporte_loso_v1.txt              -> CV por sujeto + Test final
      metricas_por_fold_loso_v1.png    -> Barras por fold
      matriz_confusion_test_v1.png     -> Confusión del sujeto test
      comparacion_folds_loso_v1.png    -> Loss y Accuracy superpuestos
      f1_por_clase_test_v1.png         -> F1 por clase del test
    """)