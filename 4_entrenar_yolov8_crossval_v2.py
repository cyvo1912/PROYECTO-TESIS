import os
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd

# ============================================================
# SCRIPT 4 V2 - ENTRENAMIENTO YOLOV8 CON CROSS-VALIDATION
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
# CONFIGURACIÓN - HIPERPARÁMETROS AJUSTADOS
# ============================================================

DATASET_RAIZ  = r"C:\Users\PC\Desktop\PRUEBA CODIGO TESIS\dataset_final"
DATASET_KFOLD = r"C:\Users\PC\Desktop\PRUEBA CODIGO TESIS\dataset_kfold_v2"

# ---- CAMBIOS RESPECTO A V1 ----
MODELO_BASE   = "yolov8s-cls.pt"    
N_FOLDS       = 5
EPOCHS        = 30                   
IMG_SIZE      = 224
BATCH_SIZE    = 32
DEVICE        = 'cpu'
LR0           = 0.005                
DROPOUT       = 0.2                  
OPTIMIZER     = 'AdamW'              
# --------------------------------

PROYECTO      = "resultados_distracciones_v2"
NOMBRE_BASE   = "yolov8s_crossval"

CLASES = [
    "hair_and_makeup",
    "phonecall_left",
    "phonecall_right",
    "reach_side",
    "safe_drive",
    "texting_left",
    "texting_right",
]

SEMILLA = 42

RESULTADOS_DIR = os.path.join(PROYECTO, NOMBRE_BASE, "resultados_crossval")
os.makedirs(RESULTADOS_DIR, exist_ok=True)

colores_fold = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]


# ============================================================
# PASO 1 - DIVIDIR CADA CLASE EN N FOLDS
# ============================================================

def crear_folds():

    print("\n" + "="*60)
    print(f"  DIVIDIENDO DATASET EN {N_FOLDS} FOLDS")
    print("="*60)

    random.seed(SEMILLA)
    folds_por_clase = {}

    for clase in CLASES:
        ruta_clase = os.path.join(DATASET_RAIZ, clase)
        imagenes   = list(Path(ruta_clase).glob("*.jpg"))
        random.shuffle(imagenes)

        folds = [[] for _ in range(N_FOLDS)]
        for i, img in enumerate(imagenes):
            folds[i % N_FOLDS].append(img)

        folds_por_clase[clase] = folds
        tamanos = [len(f) for f in folds]
        print(f"  {clase:<20s}: {len(imagenes):>5} imgs -> folds {tamanos}")

    return folds_por_clase


# ============================================================
# PASO 2 - PREPARAR CARPETAS PARA UN FOLD
# ============================================================

def preparar_fold(folds_por_clase, fold_test):

    if os.path.exists(DATASET_KFOLD):
        shutil.rmtree(DATASET_KFOLD)

    for split in ["train", "val", "test"]:
        for clase in CLASES:
            os.makedirs(os.path.join(DATASET_KFOLD, split, clase), exist_ok=True)

    random.seed(SEMILLA + fold_test)

    for clase in CLASES:
        folds = folds_por_clase[clase]

        test_imgs  = folds[fold_test]
        train_imgs = []
        for i in range(N_FOLDS):
            if i != fold_test:
                train_imgs.extend(folds[i])

        random.shuffle(train_imgs)

        n_val      = int(len(train_imgs) * 0.15)
        val_imgs   = train_imgs[:n_val]
        train_imgs = train_imgs[n_val:]

        for img in train_imgs:
            shutil.copy(str(img), os.path.join(DATASET_KFOLD, "train", clase, img.name))
        for img in val_imgs:
            shutil.copy(str(img), os.path.join(DATASET_KFOLD, "val",   clase, img.name))
        for img in test_imgs:
            shutil.copy(str(img), os.path.join(DATASET_KFOLD, "test",  clase, img.name))


# ============================================================
# PASO 3 - ENTRENAR UN FOLD
# ============================================================

def entrenar_fold(fold_num):

    nombre_run = f"{NOMBRE_BASE}_fold{fold_num}"

    print(f"\n  Entrenando Fold {fold_num} ({EPOCHS} épocas, {MODELO_BASE}, lr={LR0})...")

    model = YOLO(MODELO_BASE)
    model.train(
        data      = DATASET_KFOLD,
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
# PASO 4 - EVALUAR UN FOLD
# ============================================================

def evaluar_fold(model):

    y_true = []
    y_pred = []

    for idx_clase, clase in enumerate(CLASES):
        ruta_clase = os.path.join(DATASET_KFOLD, "test", clase)
        imagenes   = list(Path(ruta_clase).glob("*.jpg"))

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
    print("  YOLOV8 - CROSS-VALIDATION V2 (Hiperparámetros ajustados)")
    print(f"  {N_FOLDS} folds | {EPOCHS} épocas | {MODELO_BASE}")
    print(f"  LR: {LR0} | Dropout: {DROPOUT} | Optimizer: {OPTIMIZER}")
    print("="*60)

    # Dividir en folds
    folds_por_clase = crear_folds()

    resultados    = []
    todas_y_true  = []
    todas_y_pred  = []

    for fold in range(N_FOLDS):

        print("\n" + "="*60)
        print(f"  FOLD {fold} / {N_FOLDS - 1}")
        print("="*60)

        preparar_fold(folds_por_clase, fold)
        model = entrenar_fold(fold)

        print(f"  Evaluando Fold {fold} sobre su test set...")
        y_true, y_pred = evaluar_fold(model)

        prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        rec  = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1   = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        acc  = accuracy_score(y_true, y_pred)

        resultados.append({
            "fold": fold,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "accuracy": acc,
        })

        todas_y_true.extend(y_true)
        todas_y_pred.extend(y_pred)

        print(f"\n  Fold {fold} -> Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | Acc: {acc:.4f}")

    # ========================================================
    # TABLA RESUMEN
    # ========================================================

    print("\n" + "="*60)
    print("  RESUMEN CROSS-VALIDATION V2")
    print("="*60)
    print(f"\n  {'':<10s}{'Precision':>12s}{'Recall':>12s}{'F1':>12s}{'Accuracy':>12s}")
    print("  " + "-"*56)

    for r in resultados:
        print(f"  Fold {r['fold']:<5d}{r['precision']:>12.4f}{r['recall']:>12.4f}{r['f1']:>12.4f}{r['accuracy']:>12.4f}")

    prom_prec = np.mean([r["precision"] for r in resultados])
    prom_rec  = np.mean([r["recall"]    for r in resultados])
    prom_f1   = np.mean([r["f1"]        for r in resultados])
    prom_acc  = np.mean([r["accuracy"]  for r in resultados])
    std_f1    = np.std([r["f1"]         for r in resultados])

    print("  " + "-"*56)
    print(f"  {'Promedio':<10s}{prom_prec:>12.4f}{prom_rec:>12.4f}{prom_f1:>12.4f}{prom_acc:>12.4f}")
    print(f"\n  Desviación estándar F1: {std_f1:.4f}")

    # ========================================================
    # REPORTE TXT
    # ========================================================

    ruta_txt = os.path.join(RESULTADOS_DIR, "reporte_crossvalidation_v2.txt")
    with open(ruta_txt, "w", encoding="utf-8") as f:
        f.write("REPORTE CROSS-VALIDATION V2 - YOLOv8s Distracciones\n")
        f.write("="*60 + "\n")
        f.write(f"Modelo: {MODELO_BASE} | Folds: {N_FOLDS} | Épocas: {EPOCHS}\n")
        f.write(f"LR: {LR0} | Dropout: {DROPOUT} | Optimizer: {OPTIMIZER}\n")
        f.write("="*60 + "\n\n")
        f.write(f"{'':<10s}{'Precision':>12s}{'Recall':>12s}{'F1':>12s}{'Accuracy':>12s}\n")
        f.write("-"*56 + "\n")
        for r in resultados:
            f.write(f"Fold {r['fold']:<5d}{r['precision']:>12.4f}{r['recall']:>12.4f}{r['f1']:>12.4f}{r['accuracy']:>12.4f}\n")
        f.write("-"*56 + "\n")
        f.write(f"{'Promedio':<10s}{prom_prec:>12.4f}{prom_rec:>12.4f}{prom_f1:>12.4f}{prom_acc:>12.4f}\n")
        f.write(f"\nDesviación estándar F1: {std_f1:.4f}\n\n")
        f.write("\nREPORTE DETALLADO POR CLASE (todos los folds):\n")
        f.write("="*60 + "\n")
        f.write(classification_report(
            todas_y_true, todas_y_pred,
            target_names=CLASES, digits=4, zero_division=0
        ))

    print(f"\n  [OK] Reporte guardado en: {ruta_txt}")

    # ========================================================
    # GRÁFICO DE BARRAS POR FOLD
    # ========================================================

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(N_FOLDS)
    ancho = 0.25

    ax.bar(x - ancho, [r["precision"] for r in resultados], ancho, label="Precision", color="#2196F3")
    ax.bar(x,         [r["recall"]    for r in resultados], ancho, label="Recall",    color="#4CAF50")
    ax.bar(x + ancho, [r["f1"]        for r in resultados], ancho, label="F1",        color="#FF9800")

    ax.set_title("Métricas por Fold - Cross-Validation V2", fontsize=13, fontweight="bold")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {i}" for i in range(N_FOLDS)])
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    ruta_graf = os.path.join(RESULTADOS_DIR, "metricas_por_fold_v2.png")
    plt.savefig(ruta_graf, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Gráfico guardado en: {ruta_graf}")

    # ========================================================
    # MATRIZ DE CONFUSIÓN GLOBAL (NORMALIZADA + ABSOLUTA)
    # ========================================================

    todas_y_true = np.array(todas_y_true)
    todas_y_pred = np.array(todas_y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("Matriz de Confusión Global V2 (Cross-Validation)", fontsize=14, fontweight="bold")

    for ax, normalize, titulo, fmt in zip(
        axes,
        [None, "true"],
        ["Valores Absolutos", "Normalizada (%)"],
        ["d", ".2f"]
    ):
        cm = confusion_matrix(todas_y_true, todas_y_pred, normalize=normalize)
        sns.heatmap(
            cm, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=CLASES, yticklabels=CLASES,
            ax=ax, linewidths=0.5
        )
        ax.set_title(titulo, fontsize=12)
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Real")
        ax.tick_params(axis="x", rotation=35)
        ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    ruta_cm = os.path.join(RESULTADOS_DIR, "matriz_confusion_v2.png")
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

        # Curvas individuales por fold
        fig, axes = plt.subplots(N_FOLDS, 2, figsize=(14, 4 * N_FOLDS))
        fig.suptitle("Curvas de Entrenamiento por Fold - V2", fontsize=14, fontweight="bold")

        for i, (fold, df) in enumerate(csvs):
            col_train_loss = [c for c in df.columns if "train" in c and "loss" in c]
            col_val_loss   = [c for c in df.columns if "val"   in c and "loss" in c]
            epochs_x = df["epoch"] if "epoch" in df.columns else range(1, len(df) + 1)

            ax = axes[i][0]
            for col in col_train_loss:
                ax.plot(epochs_x, df[col], label="Train", linestyle="-",  color="#2196F3")
            for col in col_val_loss:
                ax.plot(epochs_x, df[col], label="Val",   linestyle="--", color="#FF5722")
            ax.set_title(f"Fold {fold} - Pérdida (Loss)", fontsize=11)
            ax.set_xlabel("Época")
            ax.set_ylabel("Loss")
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Accuracy (buscar metrics/accuracy)
            col_acc = [c for c in df.columns if "acc" in c.lower() and "top1" in c.lower()]
            ax = axes[i][1]
            for col in col_acc:
                ax.plot(epochs_x, df[col], label="Val Accuracy", linestyle="-", color="#4CAF50")
            ax.set_title(f"Fold {fold} - Accuracy", fontsize=11)
            ax.set_xlabel("Época")
            ax.set_ylabel("Accuracy")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        ruta_curvas = os.path.join(RESULTADOS_DIR, "curvas_por_fold_v2.png")
        plt.savefig(ruta_curvas, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [OK] Curvas por fold guardadas en: {ruta_curvas}")

        # Folds superpuestos
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Comparación de Folds V2 - Loss y Accuracy", fontsize=14, fontweight="bold")

        for i, (fold, df) in enumerate(csvs):
            col_val_loss = [c for c in df.columns if "val" in c and "loss" in c]
            col_acc      = [c for c in df.columns if "acc" in c.lower() and "top1" in c.lower()]
            epochs_x = df["epoch"] if "epoch" in df.columns else range(1, len(df) + 1)

            for col in col_val_loss:
                axes[0].plot(epochs_x, df[col], label=f"Fold {fold}",
                            color=colores_fold[i], linewidth=1.5)
            for col in col_acc:
                axes[1].plot(epochs_x, df[col], label=f"Fold {fold}",
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
        ruta_comp = os.path.join(RESULTADOS_DIR, "comparacion_folds_v2.png")
        plt.savefig(ruta_comp, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [OK] Comparación de folds guardada en: {ruta_comp}")

    print(f"\n{'='*60}")
    print(f"  CROSS-VALIDATION V2 COMPLETADO")
    print(f"  Resultados en: {RESULTADOS_DIR}")
    print(f"{'='*60}")
    print(f"""
  ARCHIVOS GENERADOS:
    {RESULTADOS_DIR}/
      reporte_crossvalidation_v2.txt  -> Tabla folds + reporte por clase
      metricas_por_fold_v2.png        -> Barras Precision/Recall/F1
      matriz_confusion_v2.png         -> Absoluta + Normalizada
      curvas_por_fold_v2.png          -> Loss y Accuracy por fold
      comparacion_folds_v2.png        -> Folds superpuestos
    """)
