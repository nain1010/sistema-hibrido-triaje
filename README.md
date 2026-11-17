# 🏥 Sistema Híbrido de Triaje Clínico

Sistema híbrido de apoyo a la toma de decisiones clínicas para triaje médico, combinando un **motor de reglas experto basado en conocimiento** y **modelos de Machine Learning**.

Este repositorio contiene la entrega correspondiente al **Sprint 1 (17 - 20 de Noviembre)**.

---

## 📁 Estructura del Proyecto

```plaintext
mi_proyecto_hibrido/
├── data/
│   └── processed/       # Dataset limpio y preprocesado
├── models/              # Modelos entrenados (.pkl de ML)
├── src/
│   ├── __init__.py
│   └── motor_reglas.py  # Motor de inferencia de reglas expertas
├── reglas_experto.json  # Base de conocimiento (reglas lógicas de triaje)
├── requirements.txt     # Dependencias del proyecto
├── .gitignore           # Exclusiones de Git
└── README.md            # Documentación e instrucciones
```

---

## 🩺 Variables Clínicas Evaluadas

Las variables corresponden de forma exacta a los atributos del paciente:

| Variable | Tipo | Descripción |
| :--- | :--- | :--- |
| `fiebre` | Float | Temperatura corporal en °C |
| `edad` | Int | Edad del paciente en años |
| `presion_sistolica` | Float / Int | Presión arterial sistólica (mmHg) |
| `presion_diastolica`| Float / Int | Presión arterial diastólica (mmHg) |
| `bmi` | Float | Índice de Masa Corporal (kg/m²) |
| `glucosa` | Float / Int | Nivel de glucosa en sangre (mg/dL) |
| `tos_seca` | Int | Presencia de tos seca (1: Sí, 0: No) |

---

## 📋 Base de Conocimiento (`reglas_experto.json`)

El sistema incluye 8 reglas expertas iniciales con factores de confianza:

1. **R01** (`0.92`): Fiebre alta (> 38.5 °C) con tos seca (`tos_seca == 1`) $\rightarrow$ *sospecha_infeccion_respiratoria*.
2. **R02** (`0.88`): Glucosa elevada (> 180 mg/dL) en paciente $\ge$ 60 años $\rightarrow$ *riesgo_metabolico_alto*.
3. **R03** (`0.85`): Presión sistólica crítica (> 140 mmHg) $\rightarrow$ *riesgo_hipertension*.
4. **R04** (`0.80`): IMC elevado (> 30.0) y presión sistólica (> 130 mmHg) $\rightarrow$ *riesgo_cardiovascular*.
5. **R05** (`0.70`): Fiebre moderada (> 37.5 °C) en paciente joven (< 40 años) $\rightarrow$ *infeccion_leve*.
6. **R06** (`0.75`): Glucosa limítrofe (140 a 179 mg/dL) en adultos $\rightarrow$ *precaucion_glucosa*.
7. **R07** (`0.82`): Presión diastólica elevada (> 90 mmHg) $\rightarrow$ *alerta_diastolica*.
8. **R08** (`0.65`): Fiebre (> 38.0 °C) sin tos (`tos_seca == 0`) $\rightarrow$ *fiebre_origen_desconocido*.

---

## 🚀 Instalación y Ejecución

### 1. Requisitos Previos

Tener instalado Python 3.9 o superior.

Instalar dependencias:
```bash
pip install -r requirements.txt
```

### 2. Ejecutar el Motor de Inferencia

Puedes ejecutar la prueba del motor directamente:

```bash
# Desde la raíz del proyecto
python src/motor_reglas.py

# O desde el directorio src/
cd src
python motor_reglas.py
```

### 3. Ejemplo de Uso en Código

```python
from src.motor_reglas import MotorReglas

# Inicializar motor
motor = MotorReglas("reglas_experto.json")

# Definir datos del paciente
paciente = {
    "fiebre": 39.0,
    "edad": 45,
    "presion_sistolica": 120,
    "presion_diastolica": 80,
    "bmi": 25.0,
    "glucosa": 95,
    "tos_seca": 1
}

# Evaluar
diagnosticos = motor.evaluar_paciente(paciente)

for diag in diagnosticos:
    print(f"[{diag['regla_id']}] {diag['conclusion']} ({diag['confianza'] * 100:.0f}%) - {diag['descripcion']}")
```

Salida esperada:
```plaintext
🔍 Diagnósticos generados por el Motor de Reglas:
  • [R01] sospecha_infeccion_respiratoria (Confianza: 0.92) -> Fiebre alta con tos seca sugiere infeccion respiratoria aguda
```

---

## 📌 Próximos Pasos (Sprint 2)
- Integración de modelos predictivos de Machine Learning (`.pkl`).
- Módulo de arbitraje híbrido (combinación de inferencia lógica y probabilidades ML).
- API o interfaz de usuario para triaje hospitalario.
