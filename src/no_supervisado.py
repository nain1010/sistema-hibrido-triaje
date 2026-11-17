"""
Módulo de Aprendizaje No Supervisado (Sprint 3 - Parte A):
1. K-Means:
   - Método del Codo (Elbow Method) para determinar k óptimo.
   - Evaluación por Coeficiente de Silueta (Silhouette Score).
   - Ajuste de K-Means con k=3 (Bajo, Moderado, Alto Riesgo).
   - Caracterización clínica de los centroides.
2. Self-Organizing Map (SOM):
   - Proyección topológica 2D mediante MiniSom.
   - Visualización de U-Matrix y distribución de densidades clínicas.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from minisom import MiniSom


def ejecutar_aprendizaje_no_supervisado(raiz_proyecto):
    ruta_processed = os.path.join(raiz_proyecto, "data", "processed")
    ruta_models = os.path.join(raiz_proyecto, "models")
    ruta_reports = os.path.join(raiz_proyecto, "reports")
    os.makedirs(ruta_models, exist_ok=True)
    os.makedirs(ruta_reports, exist_ok=True)

    # 1. Cargar datos de entrenamiento sin etiqueta target
    ruta_train = os.path.join(ruta_processed, "train.csv")
    if not os.path.exists(ruta_train):
        ruta_dataset = os.path.join(ruta_processed, "dataset_clean.csv")
        df_full = pd.read_csv(ruta_dataset)
        df_features = df_full.drop(columns=["target"], errors="ignore")
    else:
        df_features = pd.read_csv(ruta_train).drop(columns=["target"], errors="ignore")

    columnas_features = ["fiebre", "edad", "presion_sistolica", "presion_diastolica", "bmi", "glucosa", "tos_seca"]
    X = df_features[columnas_features].copy()

    # Cargar o ajustar scaler
    ruta_scaler = os.path.join(ruta_models, "scaler.pkl")
    if os.path.exists(ruta_scaler):
        scaler = joblib.load(ruta_scaler)
        X_scaled = scaler.transform(X)
    else:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        joblib.dump(scaler, ruta_scaler)

    print(f"Datos no supervisados cargados: {X_scaled.shape[0]} muestras, {X_scaled.shape[1]} variables.")

    # -------------------------------------------------------------
    # 2. MÉTODO DEL CODO Y SILHOUETTE PARA K-MEANS
    # -------------------------------------------------------------
    print("\n--- Evaluando Método del Codo y Coeficiente de Silueta ---")
    inercia = []
    siluetas = []
    k_range = range(2, 10)

    # Inercia para k=1
    kmeans_1 = KMeans(n_clusters=1, random_state=42, n_init=10).fit(X_scaled)
    inercia_1 = kmeans_1.inertia_
    inercias_completas = [inercia_1]
    k_completos = [1] + list(k_range)

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inercia.append(km.inertia_)
        inercias_completas.append(km.inertia_)
        sil = silhouette_score(X_scaled, labels)
        siluetas.append(sil)
        print(f"k = {k} | Inercia = {km.inertia_:.2f} | Silueta = {sil:.4f}")

    # Graficar Codo y Silueta
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(k_completos, inercias_completas, marker='o', linestyle='-', color='b')
    ax1.axvline(x=3, color='r', linestyle='--', label='k óptimo sugerido (k=3)')
    ax1.set_title("Método del Codo para K-Means (Inercia)", fontsize=12)
    ax1.set_xlabel("Número de Clústeres (k)")
    ax1.set_ylabel("Inercia (WCSS)")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    ax2.plot(list(k_range), siluetas, marker='s', linestyle='-', color='g')
    ax2.axvline(x=3, color='r', linestyle='--', label='k clínico recomendado (k=3)')
    ax2.set_title("Coeficiente de Silueta por Clúster", fontsize=12)
    ax2.set_xlabel("Número de Clústeres (k)")
    ax2.set_ylabel("Silhouette Score")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    ruta_grafico_codo = os.path.join(ruta_reports, "codo_kmeans.png")
    plt.savefig(ruta_grafico_codo, dpi=300)
    plt.close()
    print(f"Gráfico del codo guardado en: {ruta_grafico_codo}")

    # -------------------------------------------------------------
    # 3. ENTRENAR K-MEANS FINAL CON k=3 (Nivel de Triaje Clínico)
    # -------------------------------------------------------------
    k_optimo = 3
    kmeans_final = KMeans(n_clusters=k_optimo, random_state=42, n_init=15)
    cluster_labels = kmeans_final.fit_predict(X_scaled)
    joblib.dump(kmeans_final, os.path.join(ruta_models, "kmeans.pkl"))

    # Desescalar centroides para interpretación clínica en unidades originales
    centroides_originales = scaler.inverse_transform(kmeans_final.cluster_centers_)
    df_centroides = pd.DataFrame(centroides_originales, columns=columnas_features)
    df_centroides.index = [f"Cluster_{i}" for i in range(k_optimo)]

    # Contar pacientes por cluster
    conteos = pd.Series(cluster_labels).value_counts().sort_index()
    df_centroides["Num_Pacientes"] = conteos.values
    
    ruta_centroides = os.path.join(ruta_reports, "centroides_kmeans.csv")
    df_centroides.to_csv(ruta_centroides)
    print("\n--- Centroides de Clústeres en Unidades Clínicas Originales ---")
    print(df_centroides.round(2).to_string())

    # Visualización 2D mediante PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.8, edgecolors='k')
    centroides_pca = pca.transform(kmeans_final.cluster_centers_)
    plt.scatter(centroides_pca[:, 0], centroides_pca[:, 1], c='red', marker='X', s=200, label='Centroides')
    plt.title(f"Agrupamiento K-Means de Pacientes (k={k_optimo}) - Proyección PCA", fontsize=12)
    plt.xlabel(f"Componente Principal 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"Componente Principal 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.colorbar(scatter, label="Clúster Asignado")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    ruta_grafico_clusters = os.path.join(ruta_reports, "clusters_kmeans.png")
    plt.savefig(ruta_grafico_clusters, dpi=300)
    plt.close()
    print(f"Gráfico de clusters guardado en: {ruta_grafico_clusters}")

    # -------------------------------------------------------------
    # 4. MAPA AUTOORGANIZADO (SOM - SELF-ORGANIZING MAP)
    # -------------------------------------------------------------
    print("\n--- Entrenando Mapa Autoorganizado (SOM) ---")
    som_dim_x = 7
    som_dim_y = 7
    som = MiniSom(
        x=som_dim_x,
        y=som_dim_y,
        input_len=len(columnas_features),
        sigma=1.2,
        learning_rate=0.5,
        random_seed=42
    )
    som.random_weights_init(X_scaled)
    som.train_random(data=X_scaled, num_iteration=2000, verbose=False)
    joblib.dump(som, os.path.join(ruta_models, "som.pkl"))

    # U-Matrix (Matriz de Distancias Topológicas entre neuronas vecinas)
    u_matrix = som.distance_map()

    plt.figure(figsize=(9, 7))
    plt.pcolor(u_matrix.T, cmap='bone_r') # bone_r: oscuro = cercanas (clusters), claro = fronteras
    plt.colorbar(label='Distancia Euclídea Media (Fronteras Topológicas)')

    # Superponer posiciones ganadoras de los pacientes
    for idx, x_val in enumerate(X_scaled):
        w = som.winner(x_val)
        plt.plot(w[0] + 0.5, w[1] + 0.5, 'o', markerfacecolor='None',
                 markeredgecolor='crimson', markersize=6, markeredgewidth=1.2)

    plt.title("Mapa Autoorganizado (SOM) - Topología y Agrupación de Pacientes", fontsize=12)
    plt.xticks(np.arange(0.5, som_dim_x + 0.5), range(som_dim_x))
    plt.yticks(np.arange(0.5, som_dim_y + 0.5), range(som_dim_y))
    plt.xlabel("Dimensión X de la Cuadrícula")
    plt.ylabel("Dimensión Y de la Cuadrícula")
    plt.tight_layout()
    ruta_grafico_som = os.path.join(ruta_reports, "som_umatrix.png")
    plt.savefig(ruta_grafico_som, dpi=300)
    plt.close()
    print(f"Gráfico SOM U-Matrix guardado en: {ruta_grafico_som}")

    return df_centroides


if __name__ == "__main__":
    raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ejecutar_aprendizaje_no_supervisado(raiz)
