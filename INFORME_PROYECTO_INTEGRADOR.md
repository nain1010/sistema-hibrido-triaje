# 🏥 INFORME TÉCNICO FINAL: PROYECTO INTEGRADOR
## Sistema Híbrido Inteligente de Triaje Clínico Hospitalario

**Autor:** Nain  
**Repositorio:** [github.com/nain1010/sistema-hibrido-triaje](https://github.com/nain1010/sistema-hibrido-triaje)  
**Ciclo de Desarrollo:** Sprint 1 (17-20 Nov), Sprint 2 (21-24 Nov), Sprint 3 (25-28 Nov)  

---

## 📑 Resumen Ejecutivo

El presente proyecto implementa una solución de inteligencia artificial integral y robusta para la priorización y estratificación de pacientes en servicios de urgencias médicas (triaje clínico). 

La arquitectura unifica tres paradigmas de la Inteligencia Artificial:
1. **Sistemas Basados en Conocimiento (Simbolismo):** Motor de inferencia y base de hechos médicos (`reglas_experto.json`).
2. **Aprendizaje Supervisado (Conexionismo / Estadístico):** Clasificación con KNN, SVM y Redes Neuronales MLP, integrados en un sistema de arbitraje híbrido ponderado.
3. **Aprendizaje No Supervisado y por Refuerzo (Exploratorio y Prescriptivo):** Agrupamiento con K-Means y Mapas Autoorganizados de Kohonen (SOM), junto con un agente autónomo de Q-Learning que optimiza la toma de decisiones clínicas evitando categóricamente los falsos negativos.

---

## 🩺 1. Dataset y Variables Clínicas Evaluadas

El conjunto de datos procesado ([dataset_clean.csv](data/processed/dataset_clean.csv)) contiene 300 pacientes con 7 parámetros fisiológicos estandarizados:

$$\mathbf{x} = [\text{fiebre}, \text{edad}, \text{presion\_sistolica}, \text{presion\_diastolica}, \text{bmi}, \text{glucosa}, \text{tos\_seca}]^T$$

* **Fiebre:** Temperatura corporal (°C). Normal: 36.0 – 37.5 °C; Febril: $>37.5$ °C; Alta: $>38.5$ °C.
* **Edad:** Años cumplidos. Adulto mayor: $\ge 60$ años.
* **Presión Sistólica / Diastólica:** Tensión arterial en mmHg. Hipertensión: $>140/90$ mmHg.
* **BMI:** Índice de masa corporal ($kg/m^2$). Obesidad: $>30.0$.
* **Glucosa:** Concentración plasmática en ayunas/azar ($mg/dL$). Hiperglucemia: $>180$ mg/dL.
* **Tos Seca:** Variable dicotómica binaria ($1$: Presente, $0$: Ausente).
* **Target:** $1$ si el paciente amerita intervención prioritaria de triaje; $0$ si se encuentra estable o con bajo riesgo.

---

## 🧩 2. Sprint 1: Motor de Reglas Expertas (SBC)

### 2.1 Base de Conocimiento (`reglas_experto.json`)
Se definieron 8 reglas clínicas lógicas con factores de certeza/confianza ($CF \in [0.65, 0.92]$):
1. **R01 ($CF=0.92$):** $\text{Fiebre} > 38.5 \land \text{Tos Seca} = 1 \implies \text{Sospecha Infección Respiratoria Aguda}$.
2. **R02 ($CF=0.88$):** $\text{Glucosa} > 180 \land \text{Edad} \ge 60 \implies \text{Riesgo Metabólico Alto}$.
3. **R03 ($CF=0.85$):** $\text{Presión Sistólica} > 140 \implies \text{Riesgo de Hipertensión Crítica}$.
4. **R04 ($CF=0.80$):** $\text{BMI} > 30.0 \land \text{Presión Sistólica} > 130 \implies \text{Riesgo Cardiovascular}$.
5. **R05 ($CF=0.70$):** $\text{Fiebre} > 37.5 \land \text{Edad} < 40 \implies \text{Infección Leve en Adulto Joven}$.
6. **R06 ($CF=0.75$):** $140 \le \text{Glucosa} \le 179 \implies \text{Precaución / Glucosa Limítrofe}$.
7. **R07 ($CF=0.82$):** $\text{Presión Diastólica} > 90 \implies \text{Alerta de Tensión Diastólica Elevada}$.
8. **R08 ($CF=0.65$):** $\text{Fiebre} > 38.0 \land \text{Tos Seca} = 0 \implies \text{Fiebre de Origen Desconocido}$.

### 2.2 Motor de Inferencia (`src/motor_reglas.py`)
El motor evalúa condicionales compuestas y extrae las conclusiones ordenadas descendentemente por factor de confianza.

---

## 🤖 3. Sprint 2: Aprendizaje Supervisado y Fusión Híbrida

### 3.1 Modelos Evaluados
Se particionó el dataset en $80\%$ entrenamiento ($N=240$) y $20\%$ prueba ($N=60$) de forma estratificada:
* **K-Nearest Neighbors (KNN):** $k=5$ vecinos con ponderación por distancia euclídea inversa.
* **Support Vector Machine (SVM):** Kernel Gaussiano RBF ($C=1.0$, $\gamma=\text{scale}$), calibrado para salida probabilística.
* **Red Neuronal Artificial (MLPClassifier):** Arquitectura multicapa con topología $[7 \to 32 \to 16 \to 1]$, función de activación ReLU y optimizador Adam.

### 3.2 Tabla Comparativa de Métricas

$$\text{Accuracy} = \frac{VP + VN}{N}, \quad \text{Precision} = \frac{VP}{VP + FP}, \quad \text{Recall} = \frac{VP}{VP + FN}, \quad F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

| Modelo | Accuracy | Precision | Recall (Sensibilidad) | F1-Score | VP | VN | FP | FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **KNN** | 0.9667 | 0.8947 | **1.0000** | 0.9444 | 17 | 41 | 2 | **0** |
| **SVM (RBF)** | **0.9833** | **0.9444** | **1.0000** | **0.9714** | 17 | 42 | 1 | **0** |
| **Red Neuronal MLP** | **0.9833** | **0.9444** | **1.0000** | **0.9714** | 17 | 42 | 1 | **0** |

### 3.3 Justificación Clínica del Recall
> [!IMPORTANT]
> En la medicina asistencial y los servicios hospitalarios de triaje, el costo de un **Falso Negativo ($FN$)** es asimétricamente superior al de un Falso Positivo:
> * **$FN$ (Catastrófico):** Un paciente en riesgo vital o descompensación es clasificado erróneamente como sano/estable y enviado a su domicilio o demorado sin supervisión, resultando en agravamiento, shock o muerte.
> * **$FP$ (Aceptable):** Un paciente estable recibe una revisión preventiva adicional.
> 
> Todos los modelos implementados alcanzaron **$Recall = 100\%$ ($FN = 0$)**, logrando sensibilidad diagnóstica total.

### 3.4 Modelo de Fusión Híbrida (`src/sistema_hibrido.py`)
Combina la probabilidad empírica de Machine Learning con el conocimiento declarativo de las reglas clínicas:

$$\text{Score Híbrido} = (P_{\text{ML}} \times 0.60) + (CF_{\text{Reglas}} \times 0.40)$$

$$\text{Decisión} = \begin{cases} 1 \quad (\text{Prioridad Urgente}) & \text{si } \text{Score Híbrido} > 0.50 \\ 0 \quad (\text{Atención Ordinaria}) & \text{en caso contrario} \end{cases}$$

---

## 🔍 4. Sprint 3: Aprendizaje No Supervisado y Aprendizaje por Refuerzo

### 4.1 Aprendizaje No Supervisado
#### A. Método del Codo para K-Means
Al suprimir la etiqueta `target`, el algoritmo K-Means calculó la inercia intraclúster (WCSS) para $k \in [1, 9]$:
* Para $k=1$, WCSS $= 1680.0$.
* Para $k=2$, WCSS $= 1435.0$ (Silueta $= 0.1477$).
* Para $k=3$, WCSS $= 1306.7$ (Silueta $= 0.1278$).
* A partir de $k=3$, la pendiente de reducción de inercia se suaviza sensiblemente, justificando matemáticamente y clínicamente $k=3$ para triaje (Estable, Alerta, Crítico).

#### B. Centroides de los Clústeres (Unidades Clínicas Reales)
* **Clúster 0 ($N=103$):** Temperatura 38.44 °C, Tensión 138.5/83.4 mmHg, Glucosa 133.1 mg/dL, Tos seca 0%. $\to$ *Pacientes con febrícula aislada sin compromiso pulmonar.*
* **Clúster 1 ($N=73$):** Temperatura 38.78 °C, Tensión 139.8/94.9 mmHg, BMI 32.8 (obesidad), Glucosa 156.1 mg/dL, Tos seca 95%. $\to$ *Pacientes de alto riesgo sistémico y respiratorio.*
* **Clúster 2 ($N=64$):** Temperatura 38.24 °C, Tensión 129.1/77.1 mmHg, Glucosa 111.4 mg/dL, Tos seca 100%. $\to$ *Infección leve o cuadro respiratorio temprano en pacientes normotensos.*

#### C. Mapa Autoorganizado de Kohonen (SOM)
Utilizando una cuadrícula bidimensional de $7 \times 7$ neuronas con `MiniSom`, la **U-Matrix** proyecta topológicamente a los pacientes. Las zonas de valles oscuros agrupan a pacientes de características fisiológicas similares, mientras que las crestas claras marcan las transiciones de gravedad médica.

---

### 4.2 Aprendizaje por Refuerzo (Q-Learning)

#### A. Definición del Proceso de Decisión de Markov (MDP)
* **Espacio de Estados $\mathcal{S}$:**
  * $s_0$: `Estable` (Sin signos de alarma)
  * $s_1$: `Precaución` (Signos limítrofes)
  * $s_2$: `Crítico` (Hipertermia con tos, crisis hipertensiva o hiperglucemia)
* **Espacio de Acciones $\mathcal{A}$:**
  * $a_0$: `Alta a casa`
  * $a_1$: `Observación en sala`
  * $a_2$: `Urgencia inmediata`
* **Matriz de Recompensas $\mathcal{R}(s, a)$:**
  * En estado `Crítico`: Dar de alta recibe la máxima penalización de **$-50.0$** (Falso Negativo estricto); Urgencia inmediata recibe **$+20.0$**.
  * En estado `Precaución`: Observación recibe **$+15.0$**; Alta recibe **$-20.0$**.
  * En estado `Estable`: Alta recibe **$+10.0$**; Urgencia innecesaria recibe **$-15.0$**.

#### B. Ecuación de Actualización de Bellman
$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

Parámetros: $\alpha = 0.10$, $\gamma = 0.90$, exploración $\epsilon$-greedy con decaimiento de $1.0 \to 0.01$.

#### C. Matriz Q Convergida y Política Óptima ($\pi^*$)
| Estado ($s$) | Alta a casa ($a_0$) | Observación ($a_1$) | Urgencia ($a_2$) | Decisión Óptima $\pi^*(s)$ |
| :--- | :---: | :---: | :---: | :--- |
| **Estable** | **+10.00** | -0.54 | -9.77 | 🟢 **Alta a casa** |
| **Precaución** | -19.15 | **+15.00** | -4.89 | 🟡 **Observación** |
| **Crítico** | **-50.00** | -10.00 | **+20.00** | 🔴 **Urgencia inmediata** |

El agente aprendió de forma infalible a **no dar de alta a ningún paciente en estado crítico**, optimizando al mismo tiempo los recursos hospitalarios.

---

## 💻 5. Plataforma de Ejecución y Dashboard

El proyecto incluye múltiples vías de interacción:
1. **Consola Interactiva:** `python main.py` permite evaluar pacientes preestablecidos o introducir signos vitales manualmente.
2. **Dashboard Web Clínico:** `python src/app_web.py` levanta una interfaz visual en `http://localhost:8000` con selección rápida de presets y visualización en tiempo real de todos los modelos.
3. **Batería de Pruebas Unitarias:** `python -m unittest discover -s tests -p "test_*.py"` ejecuta las 14 pruebas de verificación con cobertura total.

---

## 🏁 6. Conclusiones
* La combinación de sistemas basados en reglas y modelos de Machine Learning (SVM/MLP) elimina los sesgos individuales de cada enfoque: el modelo de ML generaliza ante patrones no lineales complejos, mientras que las reglas garantizan el cumplimiento innegociable de protocolos médicos.
* Los algoritmos no supervisados (K-Means y SOM) permitieron descubrir perfiles patológicos coherentes sin necesidad de etiquetas.
* El Aprendizaje por Refuerzo demostró ser una técnica idónea para codificar funciones de costo asimétricas donde los errores tienen repercusiones humanas vitales.
