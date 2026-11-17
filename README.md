# 🏥 Sistema Híbrido de Triaje Clínico

Sistema integral de apoyo a la toma de decisiones clínicas para triaje médico hospitalario, combinando:
1. **Motor de Reglas Experto** basado en conocimiento lógico (`Sprint 1`).
2. **Aprendizaje Supervisado y Fusión Híbrida** con KNN, SVM y MLP (`Sprint 2`).
3. **Aprendizaje No Supervisado y por Refuerzo** con K-Means, SOM y Q-Learning (`Sprint 3`).

---

## 📁 Estructura del Proyecto

```plaintext
mi_proyecto_hibrido/
├── data/
│   └── processed/
│       ├── dataset_clean.csv      # Dataset clínico consolidado (300 pacientes)
│       ├── train.csv              # Partición de entrenamiento (80%, 240 pacientes)
│       └── test.csv               # Partición de evaluación (20%, 60 pacientes)
├── models/
│   ├── scaler.pkl                 # StandardScaler ajustado con train.csv
│   ├── knn.pkl                    # Modelo K-Nearest Neighbors (Sprint 2)
│   ├── svm.pkl                    # Modelo Support Vector Machine (Sprint 2)
│   ├── red_neuronal_mlp.pkl       # Red Neuronal Artificial MLP (Sprint 2)
│   ├── mejor_modelo.pkl           # Mejor modelo seleccionado (Sprint 2)
│   ├── kmeans.pkl                 # Modelo de agrupamiento K-Means k=3 (Sprint 3)
│   ├── som.pkl                    # Mapa Autoorganizado de Kohonen (Sprint 3)
│   └── q_table.pkl                # Tabla Q entrenada de triaje óptimo (Sprint 3)
├── reports/
│   ├── metricas_modelos.csv       # Métricas de clasificación (Accuracy, Precision, Recall, F1)
│   ├── centroides_kmeans.csv      # Perfiles clínicos medios por cluster
│   ├── comparativa_metricas.png   # Gráfico comparativo de modelos supervisados
│   ├── confusion_matrix_*.png     # Matrices de confusión para KNN, SVM y MLP
│   ├── codo_kmeans.png            # Método del Codo y Silhouette Score
│   ├── clusters_kmeans.png        # Distribución de clusters proyectada en PCA
│   ├── som_umatrix.png            # U-Matrix y topología de pacientes en SOM
│   └── qlearning_convergencia.png # Curva de recompensa y convergencia de Q-Learning
├── src/
│   ├── __init__.py
│   ├── motor_reglas.py            # Motor de inferencia de reglas expertas (Sprint 1)
│   ├── entrenar_evaluar.py        # Modelos supervisados y métricas (Sprint 2)
│   ├── sistema_hibrido.py         # Fusión Híbrida (ML + Reglas de Expertos) (Sprint 2)
│   ├── no_supervisado.py          # K-Means (Elbow) y Self-Organizing Map (Sprint 3)
│   └── agente_qlearning.py        # MDP y Agente de Refuerzo Q-Learning (Sprint 3)
├── tests/
│   ├── test_motor.py              # Pruebas unitarias para reglas clínicas (Sprint 1)
│   ├── test_hibrido.py            # Pruebas de integración del sistema híbrido (Sprint 2)
│   └── test_sprint3.py            # Pruebas para K-Means, SOM y Q-Learning (Sprint 3)
├── reglas_experto.json            # Base de conocimiento (8 reglas clínicas)
├── requirements.txt               # Dependencias del proyecto
├── .gitignore
└── README.md                      # Documentación completa
```

---

## 🩺 Variables Clínicas

| Variable | Tipo | Descripción |
| :--- | :--- | :--- |
| `fiebre` | Float | Temperatura corporal (°C) |
| `edad` | Int | Edad del paciente (años) |
| `presion_sistolica` | Float / Int | Presión arterial sistólica (mmHg) |
| `presion_diastolica`| Float / Int | Presión arterial diastólica (mmHg) |
| `bmi` | Float | Índice de Masa Corporal (kg/m²) |
| `glucosa` | Float / Int | Glucosa en sangre (mg/dL) |
| `tos_seca` | Int | Presencia de tos seca (1: Sí, 0: No) |
| `target` | Int | Condición clínica crítica (1: Riesgo / Prioridad, 0: Sano / Estable) |

---

## 📊 Sprint 2: Aprendizaje Supervisado y Fusión Híbrida

### Comparativa de Modelos
Evaluación en conjunto de prueba estratificado ($N=60$):

| Modelo | Accuracy | Precision | Recall (Sensibilidad) | F1-Score | VP | VN | FP | FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **KNN** ($k=5$) | 0.9667 | 0.8947 | **1.0000** | 0.9444 | 17 | 41 | 2 | **0** |
| **SVM** (RBF) | **0.9833** | **0.9444** | **1.0000** | **0.9714** | 17 | 42 | 1 | **0** |
| **Red Neuronal MLP** | **0.9833** | **0.9444** | **1.0000** | **0.9714** | 17 | 42 | 1 | **0** |

> [!IMPORTANT]
> **Relevancia del Recall en Medicina:**
> Un Falso Negativo ($FN$) significa dar de alta a un paciente enfermo de gravedad, lo cual pone en riesgo su vida. Todos los modelos alcanzaron **$FN = 0$ ($Recall = 1.0$)**, garantizando la captación de todos los casos críticos.

### Fusión Híbrida
$$\text{Score Híbrido} = (P_{\text{ML}} \times 0.6) + (\text{Confianza Máx. Reglas} \times 0.4)$$

---

## 🧠 Sprint 3: No Supervisado y Aprendizaje por Refuerzo

### Parte A: K-Means y Mapa Autoorganizado (SOM)
1. **Método del Codo (Elbow Method):**
   - Se evaluó $k \in [1, 9]$ sobre las variables estandarizadas sin la etiqueta objetivo.
   - La curva de inercia muestra una inflexión pronunciada y una estabilización coherente con las categorías clínicas de triaje en **$k = 3$** (Estable, Precaución, Crítico).
2. **Interpretación de Centroides:**
   - **Clúster 0 (Moderado / Sin tos):** Presión sistólica límite (138.5 mmHg), tos seca = 0.
   - **Clúster 1 (Riesgo Alto / Crítico):** Fiebre elevada (38.8 °C), presión alta (139.8/94.9 mmHg), IMC 32.8 (obesidad), glucosa 156 mg/dL y tos seca (95%).
   - **Clúster 2 (Bajo / Infeccioso leve):** Signos estables (129/77 mmHg), glucosa normal (111 mg/dL), tos seca aislada.
3. **SOM (Self-Organizing Map):**
   - Mediante una cuadrícula $7 \times 7$ con `MiniSom`, la U-Matrix identifica fronteras topológicas nítidas separando los perfiles graves de los perfiles asintomáticos o leves.

### Parte B: MDP y Agente Q-Learning
* **Estados ($S$):** `0: Estable`, `1: Precaución`, `2: Crítico`
* **Acciones ($A$):** `0: Alta a casa`, `1: Observación`, `2: Urgencia inmediata`
* **Sistema de Recompensas:**
  - Paciente Crítico: Mandar a casa penalizado con **$-50$** (Falso Negativo), Urgencia recompensada con **$+20$**.
  - Paciente Precaución: Observación **$+15$**, Mandar a casa **$-20$**.
  - Paciente Estable: Alta a casa **$+10$**, Urgencia innecesaria **$-15$**.

#### Tabla Q Convergida y Política Óptima ($\pi^*$):
| Estado | Alta a casa | Observación | Urgencia inmediata | Decisión Óptima $\pi^*(s)$ |
| :--- | :---: | :---: | :---: | :---: |
| **Estable** | **+10.00** | -0.54 | -9.77 | 🟢 **Alta a casa** |
| **Precaución** | -19.15 | **+15.00** | -4.89 | 🟡 **Observación** |
| **Crítico** | **-50.00** | -10.00 | **+20.00** | 🔴 **Urgencia inmediata** |

---

## 🚀 Guía de Comandos y Ejecución

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Ejecutar Sprint 1 (Motor de Reglas)
python src/motor_reglas.py

# 3. Ejecutar Sprint 2 (Entrenamiento Supervisado y Fusión)
python src/entrenar_evaluar.py
python src/sistema_hibrido.py

# 4. Ejecutar Sprint 3 (No Supervisado y Q-Learning)
python src/no_supervisado.py
python src/agente_qlearning.py

# 5. Ejecutar toda la batería de pruebas unitarias
python -m unittest discover -s tests -p "test_*.py"
```
