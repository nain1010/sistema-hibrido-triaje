# INFORME TECNICO FINAL: PROYECTO INTEGRADOR
# Sistema Multimodal e Hibrido de Triaje Clinico Hospitalario Asistido por Inteligencia Artificial

**Asignatura:** Proyecto Integrador de Inteligencia Artificial  
**Ciclo Academico:** Noviembre - Diciembre 2026  
**Defensa Oral:** Miercoles 2 de Diciembre de 2026  
**Autor:** Nain  
**Repositorio GitHub:** [github.com/nain1010/sistema-hibrido-triaje](https://github.com/nain1010/sistema-hibrido-triaje)  

---

## 1. Resumen Ejecutivo y Marco Conceptual

El triaje hospitalario en unidades de urgencias representa un cuello de botella critico para los sistemas sanitarios modernos. La clasificacion erronea de un paciente en estado grave puede derivar en secuelas irreversibles o la muerte, mientras que la sobreestimacion sistematica satura las salas de cuidados intensivos.

Este proyecto integrador materializa una arquitectura de **Inteligencia Artificial Multimodal e Hibrida** estructurada a lo largo de cuatro sprints de desarrollo tecnologico:
1. **Sprint 1 (Sistemas Basados en Conocimiento):** Motor de inferencia declarativo que procesa 8 reglas clinicas expertas (`reglas_experto.json`).
2. **Sprint 2 (Aprendizaje Supervisado y Arbitraje Hibrido):** Clasificacion con KNN, SVM y Redes Neuronales MLP, priorizando la sensibilidad diagnostica ($Recall = 100\%$) y fusionando la probabilidad empirica con las reglas clinicas.
3. **Sprint 3 (Aprendizaje No Supervisado y por Refuerzo):** Agrupamiento exploratorio mediante K-Means (justificado con el Metodo del Codo) y Mapas Autoorganizados de Kohonen (SOM), integrados con un agente de Q-Learning que penaliza severamente los falsos negativos en un proceso de decision de Markov (MDP).
4. **Sprint 4 (Deep Learning y Vision Artificial):** Red Neuronal Convolucional (CNN) en TensorFlow/Keras para la deteccion automatica de neumonia y condensaciones alveolares en radiografias de torax de $150 \times 150 \times 3$, alcanzando una fusion multimodal unificada.

---

## 2. Las 4 Fichas de Investigacion Tecnica

### Ficha 1: Sistemas Basados en Conocimiento y Motores de Inferencia Clinica
* **Concepto:** Un Sistema Basado en Conocimiento (SBC / CDSS) desacopla la base de conocimiento declarativa (hechos y reglas en formato estructurado) del motor de inferencia (algoritmo evaluador que navega las premisas mediante encadenamiento hacia adelante o hacia atras).
* **Aplicacion en el Proyecto:** Se implemento una base de reglas en JSON que evalua signos vitales estandarizados (`fiebre`, `edad`, `presion_sistolica`, `presion_diastolica`, `bmi`, `glucosa`, `tos_seca`). Cada regla asigna un factor de certeza ($CF \in [0.65, 0.92]$), emulando el juicio heuristico medico ante sindromes respiratorios o metabolicos agudos.
* **Ventaja Clinica:** Interpretabilidad y trazabilidad total: los medicos pueden auditar exactamente que condicion disparo la alerta.

---

### Ficha 2: Aprendizaje Supervisado y la Asimetria del Costo de Falsos Negativos
* **Concepto:** En clasificacion binaria medica, los errores tipo I (Falsos Positivos) y tipo II (Falsos Negativos) poseen implicaciones radicalmente asimetricas. La matriz de costos clinicos impone que:
  $$\text{Costo}(FN) \gg \text{Costo}(FP)$$
* **Justificacion Matematica del Recall:**
  $$\text{Recall} = \frac{VP}{VP + FN}$$
  Maximizar el Recall al $100\%$ ($FN = 0$) garantiza que ningun paciente con patologia grave sea catalogado erroneamente como sano o dado de alta de manera prematura.
* **Resultados Obtenidos:** SVM (Kernel RBF) y Red Neuronal MLP alcanzaron $98.33\%$ de Accuracy y **$100\%$ de Recall** ($FN = 0$) en el conjunto de prueba independiente ($N=60$).

---

### Ficha 3: Mineria No Supervisada (K-Means/SOM) y Control Optimo por Refuerzo (MDP/Q-Learning)
* **K-Means y Metodo del Codo:** Agrupacion basada en la minimizacion de la inercia intra-cluster (Within-Cluster Sum of Squares, WCSS). La curva de codo identifico $k=3$ como la particion optima:
  * **Cluster 0:** Perfil febril moderado sin tos.
  * **Cluster 1:** Perfil critico con obesidad, hipertension e hiperglucemia.
  * **Cluster 2:** Perfil leve con signos estables.
* **Mapas Autoorganizados de Kohonen (SOM):** Proyeccion no lineal bidimensional ($7 \times 7$) mediante `MiniSom`. La U-Matrix delimita fronteras topologicas visuales entre estados patologicos y basales.
* **Q-Learning y MDP:** Se modelo el triaje como un Proceso de Decision de Markov con estados $\{\text{Estable}, \text{Precaucion}, \text{Critico}\}$ y acciones $\{\text{Alta}, \text{Observacion}, \text{Urgencia}\}$. La funcion de recompensa asigna una penalizacion de **$-50.0$ al Falso Negativo** en pacientes criticos, forzando la convergencia hacia la politica optima $\pi^*(\text{Critico}) = \text{Urgencia Inmediata}$.

---

### Ficha 4: Vision Artificial y Redes Neuronales Convolucionales (CNN) en Imagenologia
* **Arquitectura:** Red convolucional multicapa basada en Keras:
  $$\text{Input}(150 \times 150 \times 3) \to \text{Conv2D}(32) \to \text{MaxPool} \to \text{Conv2D}(64) \to \text{MaxPool} \to \text{Conv2D}(128) \to \text{MaxPool} \to \text{Flatten} \to \text{Dense}(128) \to \text{Dropout}(0.5) \to \text{Dense}(1, \text{Sigmoid})$$
* **Extraccion de Patrones:** Los filtros convolucionales iniciales extraen gradientes y bordes de la caja toracica, mientras que las capas profundas detectan opacidades focales, consolidacion bibasal e infiltrados alveolares caracteristicos de neumonia.
* **Generacion de Mapas de Activacion (Grad-CAM):** Proporciona explicabilidad visual al profesional medico destacando las zonas pulmonares comprometidas.

---

## 3. Resultados y Metricas Consolidadas

### 3.1 Modelos Supervisados (Sprint 2)
| Modelo | Accuracy | Precision | Recall (Sensibilidad) | F1-Score | VP | VN | FP | FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **KNN** ($k=5$) | 0.9667 | 0.8947 | **1.0000** | 0.9444 | 17 | 41 | 2 | **0** |
| **SVM (RBF)** | **0.9833** | **0.9444** | **1.0000** | **0.9714** | 17 | 42 | 1 | **0** |
| **Red Neuronal MLP** | **0.9833** | **0.9444** | **1.0000** | **0.9714** | 17 | 42 | 1 | **0** |

---

### 3.2 Tabla Q del Agente de Refuerzo (Sprint 3)
| Estado Fisiologico | Alta a casa ($a_0$) | Observacion ($a_1$) | Urgencia Inmediata ($a_2$) | Politica Optima $\pi^*(s)$ |
| :--- | :---: | :---: | :---: | :--- |
| **Estable** | **+10.00** | -0.54 | -9.77 | **Alta a casa** |
| **Precaucion** | -19.15 | **+15.00** | -4.89 | **Observacion** |
| **Critico** | **-50.00** | -10.00 | **+20.00** | **Urgencia inmediata** |

---

### 3.3 Formula de Fusion Multimodal Global (Sprint 4)
$$\text{Score Multimodal} = (P_{\text{ML}} \times 0.35) + (CF_{\text{Reglas}} \times 0.25) + (P_{\text{CNN}} \times 0.25) + (\text{Prioridad}_{\text{MDP}} \times 0.15)$$

**Regla de Seguridad Clinica:** Si $P_{\text{CNN}} \ge 0.70$ (Neumonia confirmada) o se activan $\ge 2$ reglas de alarma vital, el triaje se eleva automaticamente a **Nivel 1 (Urgencia Inmediata / Alerta Roja)**, inhibiendo altas accidentales.

---

## 4. Etica, Privacidad y Mitigacion de Sesgos en Salud

1. **Principio de No Maleficencia y Human-in-the-Loop:** El sistema fue diseñado como una herramienta de apoyo a la toma de decisiones clinicas (CDSS) y **nunca como un sustituto autonomo del medico**. El dictamen de la IA expone siempre su fundamentacion logica, probabilidad y mapa radiologico explicable.
2. **Mitigacion de Sesgos Sociodemograficos:** Se identifico el riesgo de subdiagnostico en adultos mayores y en pacientes con patologias metabolicas atipicas. Se mitigo incorporando la variable `edad` explicitamente en las reglas expertas (R02: Hiperglucemia en adulto mayor) y aplicando muestreo estratificado en las particiones de entrenamiento.
3. **Privacidad de Datos y Confidencialidad:** Cumplimiento con los estandares internacionales HIPAA y GDPR, asi como la NOM-024-SSA3-2012 (Sistemas de Informacion de Registro Electronico para la Salud). Los registros y radiografias analizados se procesan bajo estricta disociacion y anonimizacion de identificadores directos (nombres, identificaciones, numeros de expediente).

---

## 5. Bitacora de Prompts de Inteligencia Artificial

| # | Etapa / Fecha | Herramienta IA | Prompt Utilizado | Resultado Obtenido | Ajuste Tecnico Realizado |
| :-: | :---: | :---: | :--- | :--- | :--- |
| **1** | **Sprint 1**<br>17/Nov/2026 | Google Gemini 2.0 / Antigravity | *"Genera una base de conocimiento en JSON con 8 reglas clinicas para triaje que combinen fiebre, presion arterial, glucosa, BMI y tos seca con factores de confianza realistas."* | Estructura JSON con 8 reglas que correlacionaban fiebre y comorbilidades. | Se corrigio la clave de parametro `'vlr'` por `'valor'` para asegurar consistencia con el parser y se añadieron umbrales medicos para hipertension y glucosa. |
| **2** | **Sprint 2**<br>21/Nov/2026 | Antigravity AI Assistant | *"Construye un script en scikit-learn para entrenar y comparar KNN, SVM y MLP, enfatizando por que en medicina el Recall es mas importante que la Precision."* | Codigo de entrenamiento con metricas completas y matriz de confusion ($VP, VN, FP, FN$). | Se configuro `SVC(probability=True)` y estratificacion en `train_test_split` para garantizar una distribucion identica de pacientes criticos en prueba. |
| **3** | **Sprint 3**<br>25/Nov/2026 | ChatGPT / Gemini | *"Diseña un entorno de Proceso de Decision de Markov (MDP) para triaje medico donde se penalice fuertemente dar de alta a un paciente grave."* | Matriz de recompensas estandar con valores homogeneos. | Se ajusto la recompensa del estado Critico con accion Alta a **$-50.0$** para reflejar la gravedad de un Falso Negativo e incentivar la accion Urgencia ($+20$). |
| **4** | **Sprint 3**<br>27/Nov/2026 | Antigravity AI Assistant | *"Como justificar matematicamente el numero optimo de clusteres en K-Means para agrupar pacientes sin la etiqueta target usando el metodo del codo y Silhouette Score."* | Curva de inercia y coeficientes de silueta para $k=1$ hasta $k=9$. | Se selecciono $k=3$ fundamentando que coincide con los tres niveles internacionales de triaje (Bajo, Moderado, Critico) y estabiliza la inercia WCSS. |
| **5** | **Sprint 4**<br>29/Nov/2026 | Google Gemini / Antigravity | *"Diseña una arquitectura CNN en Keras para clasificar radiografias de torax de 150x150x3 en sano vs neumonia, e integrala en una funcion multimodal."* | Codigo base secuencial con Conv2D, MaxPool, Dropout y Dense. | Se creo un modulo de inferencia que funciona tanto en Colab con GPU como localmente con un analizador convolucional de densidad y mapas de calor Grad-CAM. |

---

## 6. Guia de Preparacion para la Defensa Oral (2 de Diciembre)

### Casos Clinicos a Presentar en Vivo (15 Minutos):

* **Caso 1: Paciente Sano (Alerta Verde):** Joven de 26 años, temperatura 36.6 °C, PA 118/76 mmHg, glucosa 88 mg/dL, sin tos seca y radiografia despejada.
  * *Resultado:* 0 reglas activadas, $P(\text{Riesgo}) < 1\%$, CNN sin hallazgos ($13.5\%$), Q-Learning recomienda *Alta a casa*.
* **Caso 2: Caso Limite / Frontera (Alerta Amarilla):** Adulto de 48 años, temperatura 37.6 °C, PA 134/86 mmHg, glucosa 155 mg/dL con tos seca y radiografia con discreto infiltrado peribronquial.
  * *Resultado:* Regla R06 activada, score multimodal del $36.5\%$, Q-Learning recomienda *Observacion*. Demuestra como la fusion evita altas precipitadas.
* **Caso 3: Paciente Critico Multimodal (Alerta Roja):** Adulto mayor de 68 años con fiebre 39.4 °C, PA 162/98 mmHg, glucosa 210 mg/dL, tos seca y radiografia con condensacion densa en ambas bases pulmonares.
  * *Resultado:* 5 reglas expertas activadas, $P(\text{ML}) = 91.4\%$, CNN neumonia severa ($98.0\%$), Q-Learning indica *Urgencia Inmediata*. Dictamen final: **Alerta Roja Nivel 1**.

---

## 7. Checklist de Entrega Oficial
- [x] Repositorio de GitHub limpio y actualizado con todos los modulos: `reglas_experto.json`, `requirements.txt`, `README.md`.
- [x] Scripts de los 4 modulos: `src/motor_reglas.py`, `src/entrenar_evaluar.py`, `src/sistema_hibrido.py`, `src/no_supervisado.py`, `src/agente_qlearning.py`, `src/cnn_radiografias.py`.
- [x] Notebook interactivo para Colab: `notebooks/entrenamiento_cnn_colab.ipynb`.
- [x] Dashboard Web (FastAPI): `src/app_web.py`.
- [x] Suite de 17 pruebas unitarias sin fallos: `tests/test_*.py`.
- [x] Script de defensa en vivo con los 3 casos clinicos: `src/demo_defensa.py`.
- [x] Informe tecnico completo con las 4 fichas, etica y bitacora de IA.
