# INFORME TECNICO: PROYECTO INTEGRADOR
## Sistema Hibrido Inteligente de Triaje Clinico Hospitalario

**Autor:** Nain  
**Repositorio:** [github.com/nain1010/sistema-hibrido-triaje](https://github.com/nain1010/sistema-hibrido-triaje)  
**Ciclo de Desarrollo:** Sprint 1 (17-20 Nov), Sprint 2 (21-24 Nov), Sprint 3 (25-28 Nov)  

---

## 1. Resumen Ejecutivo

El presente proyecto implementa una solucion de inteligencia artificial integral y robusta para la priorizacion y estratificacion de pacientes en servicios de urgencias medicas (triaje clinico). 

La arquitectura unifica tres paradigmas de la Inteligencia Artificial:
1. **Sistemas Basados en Conocimiento (Simbolismo):** Motor de inferencia y base de hechos medicos (`reglas_experto.json`).
2. **Aprendizaje Supervisado (Conexionismo / Estadistico):** Clasificacion con KNN, SVM y Redes Neuronales MLP, integrados en un sistema de arbitraje hibrido ponderado.
3. **Aprendizaje No Supervisado y por Refuerzo (Exploratorio y Prescriptivo):** Agrupamiento con K-Means y Mapas Autoorganizados de Kohonen (SOM), junto con un agente autonomo de Q-Learning que optimiza la toma de decisiones clinicas evitando categoricamente los falsos negativos.

---

## 2. Dataset y Variables Clinicas Evaluadas

El conjunto de datos procesado ([dataset_clean.csv](data/processed/dataset_clean.csv)) contiene 300 pacientes con 7 parametros fisiologicos estandarizados:

$$\mathbf{x} = [\text{fiebre}, \text{edad}, \text{presion\_sistolica}, \text{presion\_diastolica}, \text{bmi}, \text{glucosa}, \text{tos\_seca}]^T$$

* **Fiebre:** Temperatura corporal (°C). Normal: 36.0 – 37.5 °C; Febril: $>37.5$ °C; Alta: $>38.5$ °C.
* **Edad:** Años cumplidos. Adulto mayor: $\ge 60$ años.
* **Presion Sistolica / Diastolica:** Tension arterial en mmHg. Hipertension: $>140/90$ mmHg.
* **BMI:** Indice de masa corporal ($kg/m^2$). Obesidad: $>30.0$.
* **Glucosa:** Concentracion plasmatica en ayunas/azar ($mg/dL$). Hiperglucemia: $>180$ mg/dL.
* **Tos Seca:** Variable dicotomica binaria ($1$: Presente, $0$: Ausente).
* **Target:** $1$ si el paciente amerita intervencion prioritaria de triaje; $0$ si se encuentra estable o con bajo riesgo.

---

## 3. Sprint 1: Motor de Reglas Expertas (SBC)

### 3.1 Base de Conocimiento (`reglas_experto.json`)
Se definieron 8 reglas clinicas logicas con factores de certeza/confianza ($CF \in [0.65, 0.92]$):
1. **R01 ($CF=0.92$):** $\text{Fiebre} > 38.5 \land \text{Tos Seca} = 1 \implies \text{Sospecha Infeccion Respiratoria Aguda}$.
2. **R02 ($CF=0.88$):** $\text{Glucosa} > 180 \land \text{Edad} \ge 60 \implies \text{Riesgo Metabolico Alto}$.
3. **R03 ($CF=0.85$):** $\text{Presion Sistolica} > 140 \implies \text{Riesgo de Hipertension Critica}$.
4. **R04 ($CF=0.80$):** $\text{BMI} > 30.0 \land \text{Presion Sistolica} > 130 \implies \text{Riesgo Cardiovascular}$.
5. **R05 ($CF=0.70$):** $\text{Fiebre} > 37.5 \land \text{Edad} < 40 \implies \text{Infeccion Leve en Adulto Joven}$.
6. **R06 ($CF=0.75$):** $140 \le \text{Glucosa} \le 179 \implies \text{Precaucion / Glucosa Limitrofe}$.
7. **R07 ($CF=0.82$):** $\text{Presion Diastolica} > 90 \implies \text{Alerta de Tension Diastolica Elevada}$.
8. **R08 ($CF=0.65$):** $\text{Fiebre} > 38.0 \land \text{Tos Seca} = 0 \implies \text{Fiebre de Origen Desconocido}$.

### 3.2 Motor de Inferencia (`src/motor_reglas.py`)
El motor evalua condicionales compuestas y extrae las conclusiones ordenadas descendentemente por factor de confianza.

---

## 4. Sprint 2: Aprendizaje Supervisado y Fusion Hibrida

### 4.1 Modelos Evaluados
Se particiono el dataset en $80\%$ entrenamiento ($N=240$) y $20\%$ prueba ($N=60$) de forma estratificada:
* **K-Nearest Neighbors (KNN):** $k=5$ vecinos con ponderacion por distancia euclidea inversa.
* **Support Vector Machine (SVM):** Kernel Gaussiano RBF ($C=1.0$, $\gamma=\text{scale}$), calibrado para salida probabilistica.
* **Red Neuronal Artificial (MLPClassifier):** Arquitectura multicapa con topologia $[7 \to 32 \to 16 \to 1]$, funcion de activacion ReLU y optimizador Adam.

### 4.2 Tabla Comparativa de Metricas

$$\text{Accuracy} = \frac{VP + VN}{N}, \quad \text{Precision} = \frac{VP}{VP + FP}, \quad \text{Recall} = \frac{VP}{VP + FN}, \quad F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

| Modelo | Accuracy | Precision | Recall (Sensibilidad) | F1-Score | VP | VN | FP | FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **KNN** | 0.9667 | 0.8947 | **1.0000** | 0.9444 | 17 | 41 | 2 | **0** |
| **SVM (RBF)** | **0.9833** | **0.9444** | **1.0000** | **0.9714** | 17 | 42 | 1 | **0** |
| **Red Neuronal MLP** | **0.9833** | **0.9444** | **1.0000** | **0.9714** | 17 | 42 | 1 | **0** |

### 4.3 Justificacion Clinica del Recall
En la medicina asistencial y los servicios hospitalarios de triaje, el costo de un **Falso Negativo ($FN$)** es asimetricamente superior al de un Falso Positivo:
* **$FN$ (Catastrofico):** Un paciente en riesgo vital o descompensacion es clasificado erroneamente como sano/estable y enviado a su domicilio o demorado sin supervision, resultando en agravamiento, shock o muerte.
* **$FP$ (Aceptable):** Un paciente estable recibe una revision preventiva adicional.

Todos los modelos implementados alcanzaron **$Recall = 100\%$ ($FN = 0$)**, logrando sensibilidad diagnostica total.

### 4.4 Modelo de Fusion Hibrida (`src/sistema_hibrido.py`)
Combina la probabilidad empirica de Machine Learning con el conocimiento declarativo de las reglas clinicas:

$$\text{Score Hibrido} = (P_{\text{ML}} \times 0.60) + (CF_{\text{Reglas}} \times 0.40)$$

$$\text{Decision} = \begin{cases} 1 \quad (\text{Prioridad Urgente}) & \text{si } \text{Score Hibrido} > 0.50 \\ 0 \quad (\text{Atencion Ordinaria}) & \text{en caso contrario} \end{cases}$$

---

## 5. Sprint 3: Aprendizaje No Supervisado y Aprendizaje por Refuerzo

### 5.1 Aprendizaje No Supervisado
#### A. Metodo del Codo para K-Means
Al suprimir la etiqueta `target`, el algoritmo K-Means calculo la inercia intracluster (WCSS) para $k \in [1, 9]$:
* Para $k=1$, WCSS $= 1680.0$.
* Para $k=2$, WCSS $= 1435.0$ (Silueta $= 0.1477$).
* Para $k=3$, WCSS $= 1306.7$ (Silueta $= 0.1278$).
* A partir de $k=3$, la pendiente de reduccion de inercia se suaviza sensiblemente, justificando matematicamente y clinicamente $k=3$ para triaje (Estable, Alerta, Critico).

#### B. Centroides de los Clusteres (Unidades Clinicas Reales)
* **Cluster 0 ($N=103$):** Temperatura 38.44 °C, Tension 138.5/83.4 mmHg, Glucosa 133.1 mg/dL, Tos seca 0%. $\to$ *Pacientes con febricula aislada sin compromiso pulmonar.*
* **Cluster 1 ($N=73$):** Temperatura 38.78 °C, Tension 139.8/94.9 mmHg, BMI 32.8 (obesidad), Glucosa 156.1 mg/dL, Tos seca 95%. $\to$ *Pacientes de alto riesgo sistemico y respiratorio.*
* **Cluster 2 ($N=64$):** Temperatura 38.24 °C, Tension 129.1/77.1 mmHg, Glucosa 111.4 mg/dL, Tos seca 100%. $\to$ *Infeccion leve o cuadro respiratorio temprano en pacientes normotensos.*

#### C. Mapa Autoorganizado de Kohonen (SOM)
Utilizando una cuadricula bidimensional de $7 \times 7$ neuronas con `MiniSom`, la **U-Matrix** proyecta topologicamente a los pacientes. Las zonas de valles oscuros agrupan a pacientes de caracteristicas fisiologicas similares, mientras que las crestas claras marcan las transiciones de gravedad medica.

---

### 5.2 Aprendizaje por Refuerzo (Q-Learning)

#### A. Definicion del Proceso de Decision de Markov (MDP)
* **Espacio de Estados $\mathcal{S}$:**
  * $s_0$: `Estable` (Sin signos de alarma)
  * $s_1$: `Precaucion` (Signos limitrofes)
  * $s_2$: `Critico` (Hipertermia con tos, crisis hipertensiva o hiperglucemia)
* **Espacio de Acciones $\mathcal{A}$:**
  * $a_0$: `Alta a casa`
  * $a_1$: `Observacion en sala`
  * $a_2$: `Urgencia inmediata`
* **Matriz de Recompensas $\mathcal{R}(s, a)$:**
  * En estado `Critico`: Dar de alta recibe la maxima penalizacion de **$-50.0$** (Falso Negativo estricto); Urgencia inmediata recibe **$+20.0$**.
  * En estado `Precaucion`: Observacion recibe **$+15.0$**; Alta recibe **$-20.0$**.
  * En estado `Estable`: Alta recibe **$+10.0$**; Urgencia innecesaria recibe **$-15.0$**.

#### B. Ecuacion de Actualizacion de Bellman
$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

Parametros: $\alpha = 0.10$, $\gamma = 0.90$, exploracion $\epsilon$-greedy con decaimiento de $1.0 \to 0.01$.

#### C. Matriz Q Convergida y Politica Optima ($\pi^*$)
| Estado ($s$) | Alta a casa ($a_0$) | Observacion ($a_1$) | Urgencia ($a_2$) | Decision Optima $\pi^*(s)$ |
| :--- | :---: | :---: | :---: | :--- |
| **Estable** | **+10.00** | -0.54 | -9.77 | **Alta a casa** |
| **Precaucion** | -19.15 | **+15.00** | -4.89 | **Observacion** |
| **Critico** | **-50.00** | -10.00 | **+20.00** | **Urgencia inmediata** |

---

## 6. Plataforma de Ejecucion y Dashboard

El proyecto incluye multiples vias de interaccion:
1. **Consola Interactiva:** `python main.py` permite evaluar pacientes preestablecidos o introducir signos vitales manualmente.
2. **Dashboard Web Clinico:** `python src/app_web.py` levanta una interfaz visual en `http://localhost:8000` con seleccion rapida de presets y visualizacion en tiempo real de todos los modelos.
3. **Bateria de Pruebas Unitarias:** `python -m unittest discover -s tests -p "test_*.py"` ejecuta las 17 pruebas de verificacion con cobertura total.

---

## 7. Conclusiones
* La combinacion de sistemas basados en reglas y modelos de Machine Learning (SVM/MLP) elimina los sesgos individuales de cada enfoque: el modelo de ML generaliza ante patrones no lineales complejos, mientras que las reglas garantizan el cumplimiento innegociable de protocolos medicos.
* Los algoritmos no supervisados (K-Means y SOM) permitieron descubrir perfiles patologicos coherentes sin necesidad de etiquetas.
* El Aprendizaje por Refuerzo demostro ser una tecnica idonea para codificar funciones de costo asimetricas donde los errores tienen repercusiones humanas vitales.
