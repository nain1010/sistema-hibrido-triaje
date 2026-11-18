# 🏥 INFORME TÉCNICO FINAL: PROYECTO INTEGRADOR
# Sistema Multimodal e Híbrido de Triaje Clínico Hospitalario Asistido por Inteligencia Artificial

**Asignatura:** Proyecto Integrador de Inteligencia Artificial  
**Ciclo Académico:** Noviembre - Diciembre 2026  
**Defensa Oral:** Miércoles 2 de Diciembre de 2026  
**Autor:** Nain  
**Repositorio GitHub:** [github.com/nain1010/sistema-hibrido-triaje](https://github.com/nain1010/sistema-hibrido-triaje)  

---

## 📑 1. Resumen Ejecutivo y Marco Conceptual

El triaje hospitalario en unidades de urgencias representa un cuello de botella crítico para los sistemas sanitarios modernos. La clasificación errónea de un paciente en estado grave puede derivar en secuelas irreversibles o la muerte, mientras que la sobreestimación sistemática satura las salas de cuidados intensivos.

Este proyecto integrador materializa una arquitectura de **Inteligencia Artificial Multimodal e Híbrida** estructurada a lo largo de cuatro sprints de desarrollo tecnológico:
1. **Sprint 1 (Sistemas Basados en Conocimiento):** Motor de inferencia declarativo que procesa 8 reglas clínicas expertas (`reglas_experto.json`).
2. **Sprint 2 (Aprendizaje Supervisado y Arbitraje Híbrido):** Clasificación con KNN, SVM y Redes Neuronales MLP, priorizando la sensibilidad diagnóstica ($Recall = 100\%$) y fusionando la probabilidad empírica con las reglas clínicas.
3. **Sprint 3 (Aprendizaje No Supervisado y por Refuerzo):** Agrupamiento exploratorio mediante K-Means (justificado con el Método del Codo) y Mapas Autoorganizados de Kohonen (SOM), integrados con un agente de Q-Learning que penaliza severamente los falsos negativos en un proceso de decisión de Markov (MDP).
4. **Sprint 4 (Deep Learning y Visión Artificial):** Red Neuronal Convolucional (CNN) en TensorFlow/Keras para la detección automática de neumonía y condensaciones alveolares en radiografías de tórax de $150 \times 150 \times 3$, alcanzando una fusión multimodal unificada.

---

## 📚 2. Las 4 Fichas de Investigación Técnica

### 🩺 Ficha 1: Sistemas Basados en Conocimiento y Motores de Inferencia Clínica
* **Concepto:** Un Sistema Basado en Conocimiento (SBC / CDSS) desacopla la base de conocimiento declarativa (hechos y reglas en formato estructurado) del motor de inferencia (algoritmo evaluador que navega las premisas mediante encadenamiento hacia adelante o hacia atrás).
* **Aplicación en el Proyecto:** Se implementó una base de reglas en JSON que evalúa signos vitales estandarizados (`fiebre`, `edad`, `presion_sistolica`, `presion_diastolica`, `bmi`, `glucosa`, `tos_seca`). Cada regla asigna un factor de certeza ($CF \in [0.65, 0.92]$), emulando el juicio heurístico médico ante síndromes respiratorios o metabólicos agudos.
* **Ventaja Clínica:** Interpretabilidad y trazabilidad total: los médicos pueden auditar exactamente qué condición disparó la alerta.

---

### 🤖 Ficha 2: Aprendizaje Supervisado y la Asimetría del Costo de Falsos Negativos
* **Concepto:** En clasificación binaria médica, los errores tipo I (Falsos Positivos) y tipo II (Falsos Negativos) poseen implicaciones radicalmente asimétricas. La matriz de costos clínicos impone que:
  $$\text{Costo}(FN) \gg \text{Costo}(FP)$$
* **Justificación Matemática del Recall:**
  $$\text{Recall} = \frac{VP}{VP + FN}$$
  Maximizar el Recall al $100\%$ ($FN = 0$) garantiza que ningún paciente con patología grave sea catalogado erróneamente como sano o dado de alta de manera prematura.
* **Resultados Obtenidos:** SVM (Kernel RBF) y Red Neuronal MLP alcanzaron $98.33\%$ de Accuracy y **$100\%$ de Recall** ($FN = 0$) en el conjunto de prueba independiente ($N=60$).

---

### 🔍 Ficha 3: Minería No Supervisada (K-Means/SOM) y Control Óptimo por Refuerzo (MDP/Q-Learning)
* **K-Means y Método del Codo:** Agrupación basada en la minimización de la inercia intra-clúster (Within-Cluster Sum of Squares, WCSS). La curva de codo identificó $k=3$ como la partición óptima:
  * **Clúster 0:** Perfil febril moderado sin tos.
  * **Clúster 1:** Perfil crítico con obesidad, hipertensión e hiperglucemia.
  * **Clúster 2:** Perfil leve con signos estables.
* **Mapas Autoorganizados de Kohonen (SOM):** Proyección no lineal bidimensional ($7 \times 7$) mediante `MiniSom`. La U-Matrix delimita fronteras topológicas visuales entre estados patológicos y basales.
* **Q-Learning y MDP:** Se modeló el triaje como un Proceso de Decisión de Markov con estados $\{\text{Estable}, \text{Precaución}, \text{Crítico}\}$ y acciones $\{\text{Alta}, \text{Observación}, \text{Urgencia}\}$. La función de recompensa asigna una penalización de **$-50.0$ al Falso Negativo** en pacientes críticos, forzando la convergencia hacia la política óptima $\pi^*(\text{Crítico}) = \text{Urgencia Inmediata}$.

---

### 🩻 Ficha 4: Visión Artificial y Redes Neuronales Convolucionales (CNN) en Imagenología
* **Arquitectura:** Red convolucional multicapa basada en Keras:
  $$\text{Input}(150 \times 150 \times 3) \to \text{Conv2D}(32) \to \text{MaxPool} \to \text{Conv2D}(64) \to \text{MaxPool} \to \text{Conv2D}(128) \to \text{MaxPool} \to \text{Flatten} \to \text{Dense}(128) \to \text{Dropout}(0.5) \to \text{Dense}(1, \text{Sigmoid})$$
* **Extracción de Patrones:** Los filtros convolucionales iniciales extraen gradientes y bordes de la caja torácica, mientras que las capas profundas detectan opacidades focales, consolidación bibasal e infiltrados alveolares característicos de neumonía.
* **Generación de Mapas de Activación (Grad-CAM):** Proporciona explicabilidad visual al profesional médico destacando las zonas pulmonares comprometidas.

---

## 📊 3. Resultados y Métricas Consolidadas

### 3.1 Modelos Supervisados (Sprint 2)
| Modelo | Accuracy | Precision | Recall (Sensibilidad) | F1-Score | VP | VN | FP | FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **KNN** ($k=5$) | 0.9667 | 0.8947 | **1.0000** | 0.9444 | 17 | 41 | 2 | **0** |
| **SVM (RBF)** | **0.9833** | **0.9444** | **1.0000** | **0.9714** | 17 | 42 | 1 | **0** |
| **Red Neuronal MLP** | **0.9833** | **0.9444** | **1.0000** | **0.9714** | 17 | 42 | 1 | **0** |

---

### 3.2 Tabla Q del Agente de Refuerzo (Sprint 3)
| Estado Fisiológico | Alta a casa ($a_0$) | Observación ($a_1$) | Urgencia Inmediata ($a_2$) | Política Óptima $\pi^*(s)$ |
| :--- | :---: | :---: | :---: | :--- |
| **Estable** | **+10.00** | -0.54 | -9.77 | 🟢 **Alta a casa** |
| **Precaución** | -19.15 | **+15.00** | -4.89 | 🟡 **Observación** |
| **Crítico** | **-50.00** | -10.00 | **+20.00** | 🔴 **Urgencia inmediata** |

---

### 3.3 Fórmula de Fusión Multimodal Global (Sprint 4)
$$\text{Score Multimodal} = (P_{\text{ML}} \times 0.35) + (CF_{\text{Reglas}} \times 0.25) + (P_{\text{CNN}} \times 0.25) + (\text{Prioridad}_{\text{MDP}} \times 0.15)$$

**Regla de Seguridad Clínica:** Si $P_{\text{CNN}} \ge 0.70$ (Neumonía confirmada) o se activan $\ge 2$ reglas de alarma vital, el triaje se eleva automáticamente a **Nivel 1 (Urgencia Inmediata / Alerta Roja)**, inhibiendo altas accidentales.

---

## ⚖️ 4. Ética, Privacidad y Mitigación de Sesgos en Salud

1. **Principio de No Maleficencia y Human-in-the-Loop:** El sistema fue diseñado como una herramienta de apoyo a la toma de decisiones clínicas (CDSS) y **nunca como un sustituto autónomo del médico**. El dictamen de la IA expone siempre su fundamentación lógica, probabilidad y mapa radiológico explicable.
2. **Mitigación de Sesgos Sociodemográficos:** Se identificó el riesgo de subdiagnóstico en adultos mayores y en pacientes con patologías metabólicas atípicas. Se mitigó incorporando la variable `edad` explícitamente en las reglas expertas (R02: Hiperglucemia en adulto mayor) y aplicando muestreo estratificado en las particiones de entrenamiento.
3. **Privacidad de Datos y Confidencialidad:** Cumplimiento con los estándares internacionales HIPAA y GDPR, así como la NOM-024-SSA3-2012 (Sistemas de Información de Registro Electrónico para la Salud). Los registros y radiografías analizados se procesan bajo estricta disociación y anonimización de identificadores directos (nombres, CURP, números de expediente).

---

## 📝 5. Bitácora de Prompts de Inteligencia Artificial

| # | Etapa / Fecha | Herramienta IA | Prompt Utilizado | Resultado Obtenido | Ajuste Técnico Realizado |
| :-: | :---: | :---: | :--- | :--- | :--- |
| **1** | **Sprint 1**<br>17/Nov/2026 | Google Gemini 2.0 / Antigravity | *"Genera una base de conocimiento en JSON con 8 reglas clínicas para triaje que combinen fiebre, presión arterial, glucosa, BMI y tos seca con factores de confianza realistas."* | Estructura JSON con 8 reglas que correlacionaban fiebre y comorbilidades. | Se corrigió la clave de parámetro `'vlr'` por `'valor'` para asegurar consistencia con el parser y se añadieron umbrales médicos para hipertensión y glucosa. |
| **2** | **Sprint 2**<br>21/Nov/2026 | Antigravity AI Assistant | *"Construye un script en scikit-learn para entrenar y comparar KNN, SVM y MLP, enfatizando por qué en medicina el Recall es más importante que la Precision."* | Código de entrenamiento con métricas completas y matriz de confusión ($VP, VN, FP, FN$). | Se configuró `SVC(probability=True)` y estratificación en `train_test_split` para garantizar una distribución idéntica de pacientes críticos en prueba. |
| **3** | **Sprint 3**<br>25/Nov/2026 | ChatGPT / Gemini | *"Diseña un entorno de Proceso de Decisión de Markov (MDP) para triaje médico donde se penalice fuertemente dar de alta a un paciente grave."* | Matriz de recompensas estándar con valores homogéneos. | Se ajustó la recompensa del estado Crítico con acción Alta a **$-50.0$** para reflejar la gravedad de un Falso Negativo e incentivar la acción Urgencia ($+20$). |
| **4** | **Sprint 3**<br>27/Nov/2026 | Antigravity AI Assistant | *"Cómo justificar matemáticamente el número óptimo de clústeres en K-Means para agrupar pacientes sin la etiqueta target usando el método del codo y Silhouette Score."* | Curva de inercia y coeficientes de silueta para $k=1$ hasta $k=9$. | Se seleccionó $k=3$ fundamentando que coincide con los tres niveles internacionales de triaje (Bajo, Moderado, Crítico) y estabiliza la inercia WCSS. |
| **5** | **Sprint 4**<br>29/Nov/2026 | Google Gemini / Antigravity | *"Diseña una arquitectura CNN en Keras para clasificar radiografías de tórax de 150x150x3 en sano vs neumonía, e intégrala en una función multimodal."* | Código base secuencial con Conv2D, MaxPool, Dropout y Dense. | Se creó un módulo de inferencia que funciona tanto en Colab con GPU como localmente con un analizador convolucional de densidad y mapas de calor Grad-CAM. |

---

## 🎤 6. Guía de Preparación para la Defensa Oral (2 de Diciembre)

### Casos Clínicos a Presentar en Vivo (15 Minutos):

* **Caso 1: Paciente Sano (Alerta Verde):** Joven de 26 años, temperatura 36.6 °C, PA 118/76 mmHg, glucosa 88 mg/dL, sin tos seca y radiografía despejada.
  * *Resultado:* 0 reglas activadas, $P(\text{Riesgo}) < 1\%$, CNN sin hallazgos ($13.5\%$), Q-Learning recomienda *Alta a casa*.
* **Caso 2: Caso Límite / Frontera (Alerta Amarilla):** Adulto de 48 años, temperatura 37.6 °C, PA 134/86 mmHg, glucosa 155 mg/dL con tos seca y radiografía con discreto infiltrado peribronquial.
  * *Resultado:* Regla R06 activada, score multimodal del $36.5\%$, Q-Learning recomienda *Observación*. Demuestra cómo la fusión evita altas precipitadas.
* **Caso 3: Paciente Crítico Multimodal (Alerta Roja):** Adulto mayor de 68 años con fiebre 39.4 °C, PA 162/98 mmHg, glucosa 210 mg/dL, tos seca y radiografía con condensación densa en ambas bases pulmonares.
  * *Resultado:* 5 reglas expertas activadas, $P(\text{ML}) = 91.4\%$, CNN neumonía severa ($98.0\%$), Q-Learning indica *Urgencia Inmediata*. Dictamen final: **Alerta Roja Nivel 1**.

---

## 📂 7. Checklist de Entrega Oficial
- [x] Repositorio de GitHub limpio y actualizado con todos los módulos: `reglas_experto.json`, `requirements.txt`, `README.md`.
- [x] Scripts de los 4 módulos: `src/motor_reglas.py`, `src/entrenar_evaluar.py`, `src/sistema_hibrido.py`, `src/no_supervisado.py`, `src/agente_qlearning.py`, `src/cnn_radiografias.py`.
- [x] Notebook interactivo para Colab: `notebooks/entrenamiento_cnn_colab.ipynb`.
- [x] Dashboard Web (FastAPI): `src/app_web.py`.
- [x] Suite de 17 pruebas unitarias sin fallos: `tests/test_*.py`.
- [x] Script de defensa en vivo con los 3 casos clínicos: `src/demo_defensa.py`.
- [x] Informe técnico completo con las 4 fichas, ética y bitácora de IA.
