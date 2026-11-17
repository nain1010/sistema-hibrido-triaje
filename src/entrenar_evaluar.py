"""
Script para entrenar y evaluar los modelos de Machine Learning (Sprint 2):
- KNN (KNeighborsClassifier)
- SVM (SVC con probability=True)
- Red Neuronal MLP (MLPClassifier)

Genera:
- Partición train.csv y test.csv (en data/processed/)
- Escalador StandardScaler (en models/scaler.pkl)
- Modelos entrenados (models/knn.pkl, models/svm.pkl, models/mlp.pkl)
- El mejor modelo seleccionado (models/mejor_modelo.pkl)
- Reporte detallado de métricas (Accuracy, Precision, Recall, F1-Score)
- Matrices de confusión (VP, VN, FP, FN)
"""

import os
import shutil
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


def preparar_datos(raiz_proyecto):
    ruta_processed = os.path.join(raiz_proyecto, "data", "processed")
    os.makedirs(ruta_processed, exist_ok=True)
    
    ruta_dataset = os.path.join(ruta_processed, "dataset_clean.csv")
    
    # Si no existe en data/processed, buscar en descargas o raíz
    if not os.path.exists(ruta_dataset):
        candidatos = [
            os.path.join(os.path.expanduser("~"), "Downloads", "dataset_clean.csv"),
            os.path.join(raiz_proyecto, "dataset_clean.csv")
        ]
        copiado = False
        for c in candidatos:
            if os.path.exists(c):
                shutil.copy(c, ruta_dataset)
                copiado = True
                print(f"Dataset copiado desde {c} a {ruta_dataset}")
                break
        if not copiado:
            raise FileNotFoundError(f"No se encontró dataset_clean.csv en las rutas esperadas.")

    df = pd.read_csv(ruta_dataset)
    print(f"Dataset cargado exitosamente. Dimensiones: {df.shape}")
    print(f"Distribución del target:\n{df['target'].value_counts(normalize=True)}")

    columnas_features = ["fiebre", "edad", "presion_sistolica", "presion_diastolica", "bmi", "glucosa", "tos_seca"]
    X = df[columnas_features]
    y = df["target"]

    # Partición 80% train / 20% test estratificada
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    df_train = pd.concat([X_train, y_train], axis=1)
    df_test = pd.concat([X_test, y_test], axis=1)

    df_train.to_csv(os.path.join(ruta_processed, "train.csv"), index=False)
    df_test.to_csv(os.path.join(ruta_processed, "test.csv"), index=False)
    print(f"Guardados train.csv ({len(df_train)} filas) y test.csv ({len(df_test)} filas) en {ruta_processed}")

    # Ajuste del escalador sobre el conjunto de entrenamiento exclusivamente
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    ruta_models = os.path.join(raiz_proyecto, "models")
    os.makedirs(ruta_models, exist_ok=True)
    joblib.dump(scaler, os.path.join(ruta_models, "scaler.pkl"))
    print(f"Scaler guardado en {os.path.join(ruta_models, 'scaler.pkl')}")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, columnas_features


def entrenar_y_evaluar(raiz_proyecto):
    X_train_s, X_test_s, y_train, y_test, scaler, features = preparar_datos(raiz_proyecto)
    
    ruta_models = os.path.join(raiz_proyecto, "models")
    ruta_reports = os.path.join(raiz_proyecto, "reports")
    os.makedirs(ruta_reports, exist_ok=True)

    # 1. Definir los 3 modelos obligatorios
    modelos = {
        "KNN": KNeighborsClassifier(n_neighbors=5, weights="distance"),
        "SVM": SVC(kernel="rbf", C=1.0, probability=True, random_state=42),
        "Red_Neuronal_MLP": MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            solver="adam",
            max_iter=1000,
            random_state=42
        )
    }

    resultados = []

    print("\n" + "="*60)
    print("INICIANDO ENTRENAMIENTO Y EVALUACIÓN DE MODELOS")
    print("="*60)

    for nombre, modelo in modelos.items():
        print(f"\n[+] Entrenando {nombre}...")
        modelo.fit(X_train_s, y_train)

        # Predicciones
        y_pred = modelo.predict(X_test_s)
        y_prob = modelo.predict_proba(X_test_s)[:, 1]

        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        # En clasificación binaria: cm[0,0]=VN, cm[0,1]=FP, cm[1,0]=FN, cm[1,1]=VP
        vn, fp, fn, vp = cm.ravel()

        # Métricas
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        # Guardar artefacto individual
        ruta_guardado = os.path.join(ruta_models, f"{nombre.lower()}.pkl")
        joblib.dump(modelo, ruta_guardado)

        resultados.append({
            "modelo": nombre,
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "VP": int(vp),
            "VN": int(vn),
            "FP": int(fp),
            "FN": int(fn)
        })

        print(f"--- Resultados para {nombre} ---")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}  <-- Métrica crítica clínica (sensibilidad)")
        print(f"F1-Score : {f1:.4f}")
        print(f"Matriz de Confusión:\n  VP: {vp} | FP: {fp}\n  FN: {fn} | VN: {vn}")

        # Gráfico de Matriz de Confusión
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Sano (0)", "Riesgo (1)"],
                    yticklabels=["Sano (0)", "Riesgo (1)"])
        plt.title(f"Matriz de Confusión - {nombre}")
        plt.xlabel("Predicción")
        plt.ylabel("Valor Real")
        plt.tight_layout()
        plt.savefig(os.path.join(ruta_reports, f"confusion_matrix_{nombre.lower()}.png"))
        plt.close()

    # Tabla resumen de resultados
    df_resultados = pd.DataFrame(resultados)
    df_resultados.to_csv(os.path.join(ruta_reports, "metricas_modelos.csv"), index=False)
    print("\n" + "="*60)
    print("RESUMEN COMPARATIVO DE MODELOS:")
    print("="*60)
    print(df_resultados.to_string(index=False))

    # Seleccionar el mejor modelo basado prioritariamente en Recall y F1-Score
    # En medicina el Recall penaliza los Falsos Negativos (riesgo de muerte/complicación)
    df_ordenado = df_resultados.sort_values(by=["recall", "f1_score", "accuracy"], ascending=False)
    mejor_nombre = df_ordenado.iloc[0]["modelo"]
    mejor_modelo = modelos[mejor_nombre]

    ruta_mejor = os.path.join(ruta_models, "mejor_modelo.pkl")
    joblib.dump(mejor_modelo, ruta_mejor)
    print(f"\n🏆 Modelo seleccionado como MEJOR MODELO: {mejor_nombre}")
    print(f"Guardado como: {ruta_mejor}")
    print(f"Métricas del mejor modelo -> Recall: {df_ordenado.iloc[0]['recall']}, F1: {df_ordenado.iloc[0]['f1_score']}, Acc: {df_ordenado.iloc[0]['accuracy']}")

    # Gráfico de barras comparativo de métricas
    plt.figure(figsize=(10, 6))
    df_plot = df_resultados.melt(id_vars=["modelo"], value_vars=["accuracy", "precision", "recall", "f1_score"],
                                var_name="Metrica", value_name="Valor")
    sns.barplot(data=df_plot, x="Metrica", y="Valor", hue="modelo", palette="Set2")
    plt.title("Comparativa de Modelos de Triaje - Sprint 2")
    plt.ylim(0, 1.1)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(ruta_reports, "comparativa_metricas.png"))
    plt.close()
    print(f"Gráficos comparativos guardados en {ruta_reports}")

    return df_resultados, mejor_nombre


if __name__ == "__main__":
    raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    entrenar_y_evaluar(raiz)
