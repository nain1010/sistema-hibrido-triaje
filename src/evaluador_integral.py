"""
Evaluador Integral de Triaje Clínico:
Combina todos los componentes de los Sprints 1, 2 y 3 en una sola interfaz:
- Sprint 1: Motor de Reglas Expertas (JSON)
- Sprint 2: Modelo Supervisado de Machine Learning (SVM/MLP) y Fusión Híbrida
- Sprint 3: Agrupamiento No Supervisado (K-Means y SOM)
- Sprint 3: Decisión Óptima de Aprendizaje por Refuerzo (Q-Learning)
"""

import os
import joblib
import numpy as np
import pandas as pd

from .motor_reglas import MotorReglas
from .sistema_hibrido import SistemaHibrido
from .agente_qlearning import EntornoTriajeMDP


class EvaluadorTriajeIntegral:
    COLUMNAS = ["fiebre", "edad", "presion_sistolica", "presion_diastolica", "bmi", "glucosa", "tos_seca"]

    def __init__(self, raiz_proyecto=None):
        if raiz_proyecto is None:
            raiz_proyecto = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.raiz = raiz_proyecto

        # Rutas de artefactos
        ruta_modelo = os.path.join(self.raiz, "models", "mejor_modelo.pkl")
        ruta_scaler = os.path.join(self.raiz, "models", "scaler.pkl")
        ruta_reglas = os.path.join(self.raiz, "reglas_experto.json")
        ruta_kmeans = os.path.join(self.raiz, "models", "kmeans.pkl")
        ruta_som = os.path.join(self.raiz, "models", "som.pkl")
        ruta_qtable = os.path.join(self.raiz, "models", "q_table.pkl")

        # Cargar componentes
        self.sistema_hibrido = SistemaHibrido(ruta_modelo, ruta_scaler, ruta_reglas)
        self.scaler = joblib.load(ruta_scaler)
        self.kmeans = joblib.load(ruta_kmeans)
        self.som = joblib.load(ruta_som)
        self.q_table = joblib.load(ruta_qtable)
        self.entorno_mdp = EntornoTriajeMDP()

    def evaluar_paciente(self, paciente_dict):
        """
        Evalúa un paciente a través de los 3 subsistemas inteligentes.
        """
        # 1. Evaluación del Sistema Híbrido (Sprint 1 + Sprint 2)
        df_paciente = pd.DataFrame([paciente_dict])[self.COLUMNAS]
        res_hibrido = self.sistema_hibrido.evaluar_paciente_completo(paciente_dict, df_paciente)

        # 2. Evaluación No Supervisada (Sprint 3 Parte A)
        x_scaled = self.scaler.transform(df_paciente)
        cluster_kmeans = int(self.kmeans.predict(x_scaled)[0])
        coordenada_som = [int(c) for c in self.som.winner(x_scaled[0])]

        descripcion_clusters = {
            0: "Perfil Febril Moderado / Sin tos",
            1: "Perfil Crítico / Infección con comorbilidades múltiples",
            2: "Perfil Leve / Signos estables con tos aislada"
        }

        # 3. Evaluación por Refuerzo (Sprint 3 Parte B)
        estado_clinico = self.entorno_mdp._clasificar_estado_clinico(paciente_dict)
        nombre_estado = self.entorno_mdp.NOMBRES_ESTADOS[estado_clinico]
        accion_optima_idx = int(np.argmax(self.q_table[estado_clinico]))
        nombre_accion = self.entorno_mdp.NOMBRES_ACCIONES[accion_optima_idx]
        valores_q = [round(float(v), 2) for v in self.q_table[estado_clinico]]

        # Consolidación de resultado
        return {
            "datos_ingresados": paciente_dict,
            "sprint_1_reglas": {
                "reglas_activadas": res_hibrido["reglas_activadas"],
                "confianza_maxima": res_hibrido["confianza_reglas"],
                "detalle_reglas": res_hibrido["detalle_reglas"]
            },
            "sprint_2_supervisado_hibrido": {
                "prediccion_ml": res_hibrido["prediccion_ml"],
                "probabilidad_riesgo_ml": round(res_hibrido["probabilidad_ml"], 4),
                "score_hibrido": round(res_hibrido["score_hibrido"], 4),
                "decision_final_triaje": "URGENCIA / PRIORITARIO" if res_hibrido["decision_final_triaje"] == 1 else "NO URGENTE / ESTABLE"
            },
            "sprint_3_no_supervisado": {
                "cluster_kmeans": cluster_kmeans,
                "cluster_interpretacion": descripcion_clusters.get(cluster_kmeans, "Desconocido"),
                "coordenadas_som": coordenada_som
            },
            "sprint_3_refuerzo": {
                "estado_mdp": nombre_estado,
                "accion_optima_qlearning": nombre_accion,
                "valores_q_estado": {
                    "Alta a casa": valores_q[0],
                    "Observación": valores_q[1],
                    "Urgencia inmediata": valores_q[2]
                }
            }
        }
