# 🏥 Sistema Multimodal e Híbrido de Triaje Clínico

Sistema integral de apoyo a la toma de decisiones clínicas para triaje médico hospitalario asistido por Inteligencia Artificial, combinando:
1. **Sprint 1 (Reglas Expertas):** Motor de inferencia basado en lógica médica y factores de certeza (`reglas_experto.json`).
2. **Sprint 2 (Aprendizaje Supervisado y Arbitraje Híbrido):** KNN, SVM y MLP con **$100\%$ de Recall** ($FN = 0$) y fusión de probabilidades.
3. **Sprint 3 (Aprendizaje No Supervisado y por Refuerzo):** Agrupamiento K-Means ($k=3$), Mapas SOM ($7 \times 7$) y agente Q-Learning con penalización severa a falsos negativos ($-50$).
4. **Sprint 4 y Cierre (Deep Learning y Visión Artificial):** Red Convolucional (CNN) para detección de neumonía en radiografías de tórax ($150 \times 150 \times 3$) y fusión multimodal completa.

---

## 📁 Estructura del Repositorio

```plaintext
mi_proyecto_hibrido/
├── data/
│   ├── processed/
│   │   ├── dataset_clean.csv      # Dataset clínico consolidado (300 pacientes)
│   │   ├── train.csv              # Partición de entrenamiento (80%, 240 pacientes)
│   │   └── test.csv               # Partición de evaluación (20%, 60 pacientes)
│   └── radiografias/              # Imágenes radiográficas de tórax simuladas y calibradas
│       ├── paciente_sano.png
│       ├── paciente_limite.png
│       └── paciente_critico_neumonia.png
├── models/
│   ├── scaler.pkl                 # StandardScaler ajustado con train.csv
│   ├── knn.pkl                    # Modelo K-Nearest Neighbors (Sprint 2)
│   ├── svm.pkl                    # Modelo Support Vector Machine (Sprint 2)
│   ├── red_neuronal_mlp.pkl       # Red Neuronal Artificial MLP (Sprint 2)
│   ├── mejor_modelo.pkl           # Mejor modelo supervisado (Sprint 2)
│   ├── kmeans.pkl                 # Modelo K-Means k=3 (Sprint 3)
│   ├── som.pkl                    # Mapa Autoorganizado de Kohonen (Sprint 3)
│   └── q_table.pkl                # Tabla Q entrenada de triaje óptimo (Sprint 3)
├── notebooks/
│   └── entrenamiento_cnn_colab.ipynb # Notebook optimizado para GPU en Google Colab (Sprint 4)
├── reports/
│   ├── metricas_modelos.csv       # Métricas de clasificación (Accuracy, Precision, Recall, F1)
│   ├── centroides_kmeans.csv      # Perfiles clínicos medios por cluster
│   ├── comparativa_metricas.png   # Gráfico comparativo de modelos supervisados
│   ├── confusion_matrix_*.png     # Matrices de confusión para KNN, SVM y MLP
│   ├── codo_kmeans.png            # Método del Codo y Silhouette Score
│   ├── clusters_kmeans.png        # Distribución de clusters proyectada en PCA
│   ├── som_umatrix.png            # U-Matrix y topología de pacientes en SOM
│   ├── qlearning_convergencia.png # Curva de recompensa de Q-Learning
│   └── demo_caso_*_heatmap.png    # Mapas de activación radiológica Grad-CAM
├── src/
│   ├── __init__.py
│   ├── motor_reglas.py            # Motor de inferencia de reglas expertas (Sprint 1)
│   ├── entrenar_evaluar.py        # Modelos supervisados y métricas (Sprint 2)
│   ├── sistema_hibrido.py         # Fusión Híbrida (ML + Reglas de Expertos) (Sprint 2)
│   ├── no_supervisado.py          # K-Means (Elbow) y Self-Organizing Map (Sprint 3)
│   ├── agente_qlearning.py        # MDP y Agente de Refuerzo Q-Learning (Sprint 3)
│   ├── cnn_radiografias.py        # Visión Artificial con CNN para radiografías (Sprint 4)
│   ├── evaluador_integral.py      # Orquestador y Fusión Multimodal Global (Sprints 1-4)
│   ├── demo_defensa.py            # Demostración en vivo de los 3 casos para la defensa oral
│   └── app_web.py                 # Dashboard Web interactivo en FastAPI
├── tests/
│   ├── test_motor.py              # Pruebas unitarias para reglas clínicas (Sprint 1)
│   ├── test_hibrido.py            # Pruebas de integración del sistema híbrido (Sprint 2)
│   ├── test_sprint3.py            # Pruebas para K-Means, SOM y Q-Learning (Sprint 3)
│   └── test_sprint4.py            # Pruebas para CNN y evaluación multimodal (Sprint 4)
├── tools/
│   └── generar_reporte_html.py    # Generador de reporte imprimible HTML / PDF
├── INFORME_TECNICO_FINAL_PROYECTO.pdf # Informe técnico formal (máx. 12 páginas, 6 páginas)
├── INFORME_TECNICO_FINAL_PROYECTO.md  # Fuente Markdown del informe técnico con 4 fichas y bitácora
├── reglas_experto.json            # Base de conocimiento (8 reglas clínicas)
├── requirements.txt               # Dependencias del proyecto
├── main.py                        # Menú interactivo principal por terminal
├── .gitignore
└── README.md                      # Documentación completa
```

---

## 📋 Checklist Oficial para la Entrega (2 de Diciembre)

- [x] **Repositorio de GitHub limpio** con `reglas_experto.json`, `requirements.txt`, `README.md` estructurado y los scripts de los 4 módulos funcionales.
- [x] **Informe Técnico en PDF** ([INFORME_TECNICO_FINAL_PROYECTO.pdf](INFORME_TECNICO_FINAL_PROYECTO.pdf)) de 6 páginas conteniendo:
  - Marco conceptual y arquitectura multimodal.
  - Las 4 Fichas de Investigación Técnica completas.
  - Gráficas de métricas (matrices de confusión, método del codo, U-Matrix y curvas de convergencia).
  - Apartado de Ética, Privacidad y Mitigación de Sesgos en Salud.
  - Bitácora de Prompts de IA con los 5 casos de apoyo técnico documentados.
- [x] **Todo probado y listo para correr sin errores:**
  - 17 pruebas unitarias automatizadas con $100\%$ de éxito.
  - Script directo para la defensa oral: `python src/demo_defensa.py`.
  - Dashboard Web accesible vía navegador: `python src/app_web.py`.

---

## 🩺 Los 4 Módulos Tecnológicos Integrados

### 1️⃣ Sprint 1: Motor de Reglas Expertas (`reglas_experto.json`)
* 8 reglas clínicas heurísticas que evalúan temperatura, presión arterial, glucosa, edad, BMI y tos seca con factores de certeza ($CF = 0.65 - 0.92$).
* Ejecución: `python src/motor_reglas.py`.

### 2️⃣ Sprint 2: Aprendizaje Supervisado y Arbitraje Híbrido
* Comparación de KNN, SVM y Red Neuronal MLP sobre 300 pacientes (partición estratificada 80/20).
* **Sensibilidad Clínica:** Todos los modelos alcanzaron **$Recall = 1.0000$ ($FN = 0$)**, previniendo que pacientes críticos sean desatendidos.
* Ejecución: `python src/entrenar_evaluar.py`.

### 3️⃣ Sprint 3: No Supervisado (K-Means / SOM) y Refuerzo (Q-Learning)
* **K-Means:** Número óptimo $k=3$ justificado con el Método del Codo y Coeficiente de Silueta.
* **SOM:** Matriz U-Matrix en cuadrícula $7 \times 7$ con `MiniSom`.
* **Q-Learning:** MDP con penalización crítica de **$-50.0$ a falsos negativos**. Política óptima convergida infalible.
* Ejecución: `python src/no_supervisado.py` y `python src/agente_qlearning.py`.

### 4️⃣ Sprint 4: Red Convolucional (CNN) y Fusión Multimodal
* Análisis de radiografías de tórax ($150 \times 150 \times 3$) detectando consolidación alveolar e infiltrados neumónicos.
* **Fórmula de Fusión Multimodal:**
  $$\text{Score Multimodal} = (P_{\text{ML}} \times 0.35) + (CF_{\text{Reglas}} \times 0.25) + (P_{\text{CNN}} \times 0.25) + (\text{Prioridad}_{\text{MDP}} \times 0.15)$$
* Ejecución: `python src/cnn_radiografias.py`.

---

## 🚀 Guía de Ejecución Rápida

### 1. Ejecutar Demostración en Vivo para la Defensa Oral (15 Minutos)
Ejecuta los 3 casos clínicos obligatorios con explicaciones técnicas para el jurado:
```bash
python src/demo_defensa.py
```

### 2. Iniciar el Dashboard Web Interactivo
```bash
python src/app_web.py
```
Abre en tu navegador [http://localhost:8000](http://localhost:8000).

### 3. Menú Interactivo General
```bash
python main.py
```

### 4. Ejecutar la Batería Completa de Pruebas Unitarias
```bash
python -m unittest discover -s tests -p "test_*.py"
```
*(17 pruebas unitarias automáticas cubriendo los 4 módulos).*
