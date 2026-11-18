"""
Evaluador Integral Multimodal de Triaje Clínico:
Combina los 4 mundos tecnológicos (Sprints 1, 2, 3 y 4):
1. Sprint 1: Motor de Reglas Expertas (JSON)
2. Sprint 2: Modelo Supervisado de Machine Learning (SVM/MLP) y Fusión Híbrida
3. Sprint 3: Agrupamiento No Supervisado (K-Means y SOM) y Aprendizaje por Refuerzo (Q-Learning)
4. Sprint 4: Redes Neuronales Convolucionales (CNN) para Análisis de Radiografías de Tórax
"""

import os
import joblib
import numpy as np
import pandas as pd

try:
    from .motor_reglas import MotorReglas
    from .sistema_hibrido import SistemaHibrido
    from .agente_qlearning import EntornoTriajeMDP
    from .cnn_radiografias import ClasificadorCNNRadiografias
except ImportError:
    try:
        from motor_reglas import MotorReglas
        from sistema_hibrido import SistemaHibrido
        from agente_qlearning import EntornoTriajeMDP
        from cnn_radiografias import ClasificadorCNNRadiografias
    except ImportError:
        from src.motor_reglas import MotorReglas
        from src.sistema_hibrido import SistemaHibrido
        from src.agente_qlearning import EntornoTriajeMDP
        from src.cnn_radiografias import ClasificadorCNNRadiografias


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
        ruta_cnn_h5 = os.path.join(self.raiz, "models", "modelo_cnn_neumonia.h5")

        # Cargar componentes
        self.sistema_hibrido = SistemaHibrido(ruta_modelo, ruta_scaler, ruta_reglas)
        self.scaler = joblib.load(ruta_scaler)
        self.kmeans = joblib.load(ruta_kmeans)
        self.som = joblib.load(ruta_som)
        self.q_table = joblib.load(ruta_qtable)
        self.entorno_mdp = EntornoTriajeMDP()
        self.clasificador_cnn = ClasificadorCNNRadiografias(ruta_cnn_h5)

    def evaluar_paciente(self, paciente_dict, ruta_imagen_radiografia=None, guardar_heatmap_en=None):
        """
        Evalúa un paciente a través de los 4 subsistemas inteligentes (Multimodal).
        """
        # 1. Sprint 1 y 2: Tabular + Reglas + ML
        df_paciente = pd.DataFrame([paciente_dict])[self.COLUMNAS]
        res_hibrido = self.sistema_hibrido.evaluar_paciente_completo(paciente_dict, df_paciente)

        # 2. Sprint 3A: No Supervisado (K-Means y SOM)
        x_scaled = self.scaler.transform(df_paciente)
        cluster_kmeans = int(self.kmeans.predict(x_scaled)[0])
        coordenada_som = [int(c) for c in self.som.winner(x_scaled[0])]

        descripcion_clusters = {
            0: "Perfil Febril Moderado / Sin tos",
            1: "Perfil Crítico / Infección con comorbilidades múltiples",
            2: "Perfil Leve / Signos estables con tos aislada"
        }

        # 3. Sprint 3B: Aprendizaje por Refuerzo (MDP y Q-Learning)
        estado_clinico = self.entorno_mdp._clasificar_estado_clinico(paciente_dict)
        nombre_estado = self.entorno_mdp.NOMBRES_ESTADOS[estado_clinico]
        accion_optima_idx = int(np.argmax(self.q_table[estado_clinico]))
        nombre_accion = self.entorno_mdp.NOMBRES_ACCIONES[accion_optima_idx]
        valores_q = [round(float(v), 2) for v in self.q_table[estado_clinico]]

        # 4. Sprint 4: Deep Learning con Red Convolucional (CNN) en Radiografía
        res_cnn = None
        prob_cnn = 0.0
        if ruta_imagen_radiografia and os.path.exists(ruta_imagen_radiografia):
            res_cnn = self.clasificador_cnn.analizar_radiografia(
                ruta_imagen_radiografia,
                guardar_heatmap_en=guardar_heatmap_en
            )
            prob_cnn = res_cnn["probabilidad_neumonia"]

        # 5. FUSIÓN MULTIMODAL GLOBAL (Los 4 Mundos)
        prob_ml = res_hibrido["probabilidad_ml"]
        conf_reglas = res_hibrido["confianza_reglas"]
        factor_mdp = 1.0 if estado_clinico == 2 else (0.5 if estado_clinico == 1 else 0.0)

        if res_cnn is not None:
            # Ponderación Multimodal: 35% ML Tabular, 25% Reglas, 25% Imagen CNN, 15% Agente MDP
            score_multimodal = (prob_ml * 0.35) + (conf_reglas * 0.25) + (prob_cnn * 0.25) + (factor_mdp * 0.15)
            
            # Alerta Roja de Seguridad Clínica:
            # Si la CNN detecta neumonía severa (prob >= 0.70) o las reglas activan R01/R02, es urgencia médica
            if prob_cnn >= 0.70 or res_hibrido["reglas_activadas"] >= 2 or score_multimodal > 0.50:
                decision_final = "URGENCIA INMEDIATA / PRIORITARIO (ALERTA ROJA)"
                nivel_prioridad = 1
            elif score_multimodal > 0.30 or estado_clinico == 1 or prob_cnn >= 0.35:
                decision_final = "OBSERVACIÓN Y MONITOREO (ALERTA AMARILLA)"
                nivel_prioridad = 2
            else:
                decision_final = "ATENCIÓN AMBULATORIA / ESTABLE (ALERTA VERDE)"
                nivel_prioridad = 3
        else:
            score_multimodal = res_hibrido["score_hibrido"]
            decision_final = "URGENCIA / PRIORITARIO" if res_hibrido["decision_final_triaje"] == 1 else "NO URGENTE / ESTABLE"
            nivel_prioridad = 1 if res_hibrido["decision_final_triaje"] == 1 else 3

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
                "score_hibrido_tabular": round(res_hibrido["score_hibrido"], 4)
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
            },
            "sprint_4_vision_cnn": res_cnn,
            "dictamen_global_multimodal": {
                "score_multimodal": round(score_multimodal, 4),
                "decision_final_triaje": decision_final,
                "nivel_prioridad": nivel_prioridad,
                "justificacion_clinica": self._generar_justificacion(res_hibrido, res_cnn, nombre_accion)
            }
        }

    def _generar_justificacion(self, res_hibrido, res_cnn, accion_mdp):
        partes = []
        if res_hibrido["reglas_activadas"] > 0:
            partes.append(f"{res_hibrido['reglas_activadas']} regla(s) experta(s) activadas")
        if res_hibrido["probabilidad_ml"] > 0.5:
            partes.append(f"alta probabilidad estadística de riesgo ML ({res_hibrido['probabilidad_ml']*100:.1f}%)")
        if res_cnn:
            partes.append(f"radiografía con {res_cnn['dictamen_radiologico']}")
        partes.append(f"política de refuerzo indicando '{accion_mdp}'")
        return "Dictamen fundamentado en: " + "; ".join(partes) + "."
