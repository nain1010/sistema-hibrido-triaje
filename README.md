# 🏥 Sistema Híbrido de Triaje Clínico

Sistema híbrido de apoyo a la toma de decisiones clínicas para triaje médico, combinando un **motor de reglas experto basado en conocimiento** y **modelos de Machine Learning supervisados**.

Este repositorio contiene los entregables del **Sprint 1 (17 - 20 de Noviembre)** y **Sprint 2 (21 - 24 de Noviembre)**.

---

## 📁 Estructura del Proyecto

```plaintext
mi_proyecto_hibrido/
├── data/
│   └── processed/
│       ├── dataset_clean.csv # Dataset clínico original (300 pacientes)
│       ├── train.csv         # Conjunto de entrenamiento (80%, 240 pacientes)
│       └── test.csv          # Conjunto de evaluación (20%, 60 pacientes)
├── models/
│   ├── scaler.pkl            # StandardScaler ajustado con train.csv
│   ├── knn.pkl               # Modelo K-Nearest Neighbors entrenado
│   ├── svm.pkl               # Modelo Support Vector Machine (RBF)
│   ├── red_neuronal_mlp.pkl  # Red Neuronal Artificial (MLPClassifier)
│   └── mejor_modelo.pkl      # Mejor modelo seleccionado (SVM / MLP)
├── reports/
│   ├── metricas_modelos.csv  # Tabla comparativa con VP, VN, FP, FN y métricas
│   ├── comparativa_metricas.png
│   ├── confusion_matrix_knn.png
│   ├── confusion_matrix_svm.png
│   └── confusion_matrix_red_neuronal_mlp.png
├── src/
│   ├── __init__.py
│   ├── motor_reglas.py       # Motor de inferencia de reglas expertas (Sprint 1)
│   ├── entrenar_evaluar.py   # Entrenamiento, evaluación y reportes ML (Sprint 2)
│   └── sistema_hibrido.py    # Fusión híbrida ML + Reglas Expertas (Sprint 2)
├── tests/
│   ├── test_motor.py         # Pruebas unitarias para reglas clínicas
│   └── test_hibrido.py       # Pruebas de integración del sistema híbrido
├── reglas_experto.json       # Base de conocimiento (8 reglas clínicas)
├── requirements.txt          # Dependencias del proyecto
├── .gitignore
└── README.md                 # Documentación e instrucciones
```

---

## 🩺 Variables Clínicas Evaluadas

Las variables corresponden a los atributos clínicos del paciente:

| Variable | Tipo | Descripción |
| :--- | :--- | :--- |
| `fiebre` | Float | Temperatura corporal en °C |
| `edad` | Int | Edad del paciente en años |
| `presion_sistolica` | Float / Int | Presión arterial sistólica (mmHg) |
| `presion_diastolica`| Float / Int | Presión arterial diastólica (mmHg) |
| `bmi` | Float | Índice de Masa Corporal (kg/m²) |
| `glucosa` | Float / Int | Nivel de glucosa en sangre (mg/dL) |
| `tos_seca` | Int | Presencia de tos seca (1: Sí, 0: No) |
| `target` | Int | Condición clínica crítica (1: Riesgo / Prioridad, 0: Sano / Estable) |

---

## 📊 Sprint 2: Comparativa de Modelos Supervisados

Se entrenaron y evaluaron tres modelos con una partición estratificada 80/20 ($N=60$ en prueba):

| Modelo | Accuracy | Precision | Recall (Sensibilidad) | F1-Score | VP | VN | FP | FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **KNN** ($k=5$) | 0.9667 | 0.8947 | **1.0000** | 0.9444 | 17 | 41 | 2 | **0** |
| **SVM** (Kernel RBF) | **0.9833** | **0.9444** | **1.0000** | **0.9714** | 17 | 42 | 1 | **0** |
| **Red Neuronal MLP** | **0.9833** | **0.9444** | **1.0000** | **0.9714** | 17 | 42 | 1 | **0** |

> [!IMPORTANT]
> ### 💡 Justificación Clínica del Recall en Medicina
> En el contexto de triaje clínico hospitalario, el **Recall (Sensibilidad)** es la métrica de mayor trascendencia crítica:
> - Un **Falso Negativo ($FN$)** representa catalogar a un paciente en condición de riesgo grave como "sano" o "estable", privándolo de atención inmediata, lo que acarrea consecuencias fatales o secuelas irreversibles.
> - Un **Falso Positivo ($FP$)** simplemente implica realizar una segunda evaluación o examen preventivo adicional al paciente.
> - **Logro:** Todos los modelos obtuvieron **$FN = 0$ ($Recall = 1.0000$)**, garantizando que el 100% de los pacientes en riesgo fueron oportunamente detectados.

---

## 🔀 Fusión Híbrida (`sistema_hibrido.py`)

La toma de decisiones de triaje integra el conocimiento empírico de los datos (Machine Learning) con el conocimiento heurístico médico (Reglas de Expertos):

$$\text{Score Híbrido} = (P_{\text{ML}} \times 0.6) + (\text{Confianza Máx. Reglas} \times 0.4)$$

$$\text{Decisión Final} = \begin{cases} 1 & \text{si } \text{Score Híbrido} > 0.5 \quad (\text{Prioridad de Triaje / Riesgo Alto}) \\ 0 & \text{en caso contrario} \quad (\text{Atención Ordinaria / Estable}) \end{cases}$$

---

## 🚀 Guía de Ejecución

### 1. Instalación de Dependencias

```bash
pip install -r requirements.txt
```

### 2. Re-entrenar y Evaluar Modelos

```bash
python src/entrenar_evaluar.py
```

### 3. Ejecutar el Sistema Híbrido

```bash
python src/sistema_hibrido.py
```

Ejemplo de salida en consola:
```plaintext
🏥 Resultado del Sistema Híbrido:
  • prediccion_ml: 1
  • probabilidad_ml: 0.9774
  • reglas_activadas: 5
  • confianza_reglas: 0.92
  • score_hibrido: 0.9544
  • decision_final_triaje: 1
  • detalle_reglas: [
      • [R01] sospecha_infeccion_respiratoria (Confianza: 0.92)
      • [R02] riesgo_metabolico_alto (Confianza: 0.88)
      • [R03] riesgo_hipertension (Confianza: 0.85)
      • [R07] alerta_diastolica (Confianza: 0.82)
      • [R04] riesgo_cardiovascular (Confianza: 0.80)
    ]
```

### 4. Ejecutar Suite de Pruebas Unitarias

```bash
python -m unittest discover -s tests -p "test_*.py"
```
