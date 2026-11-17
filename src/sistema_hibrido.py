import os
import joblib
import pandas as pd

try:
    from .motor_reglas import MotorReglas
except ImportError:
    try:
        from motor_reglas import MotorReglas
    except ImportError:
        from src.motor_reglas import MotorReglas


def _resolver_ruta(ruta):
    """
    Resuelve rutas relativas buscando en el directorio de trabajo actual,
    en el directorio del script o en la raíz del proyecto.
    """
    if os.path.isabs(ruta) and os.path.exists(ruta):
        return ruta
    if os.path.exists(ruta):
        return os.path.abspath(ruta)
    
    dir_script = os.path.dirname(os.path.abspath(__file__))
    candidato_script = os.path.normpath(os.path.join(dir_script, ruta))
    if os.path.exists(candidato_script):
        return candidato_script
        
    candidato_raiz = os.path.normpath(os.path.join(dir_script, "..", ruta))
    if os.path.exists(candidato_raiz):
        return candidato_raiz

    candidato_basename = os.path.normpath(os.path.join(dir_script, "..", os.path.basename(ruta)))
    if os.path.exists(candidato_basename):
        return candidato_basename

    return ruta


class SistemaHibrido:
    COLUMNAS_MODELO = [
        "fiebre", "edad", "presion_sistolica", "presion_diastolica",
        "bmi", "glucosa", "tos_seca"
    ]

    def __init__(self, ruta_modelo, ruta_scaler, ruta_reglas):
        # Resolver rutas de artefactos
        ruta_modelo_res = _resolver_ruta(ruta_modelo)
        ruta_scaler_res = _resolver_ruta(ruta_scaler)
        ruta_reglas_res = _resolver_ruta(ruta_reglas)

        if not os.path.exists(ruta_modelo_res):
            raise FileNotFoundError(f"No se encontró el modelo ML en: {ruta_modelo}")
        if not os.path.exists(ruta_scaler_res):
            raise FileNotFoundError(f"No se encontró el escalador en: {ruta_scaler}")

        # Cargar artefactos de ML
        self.modelo = joblib.load(ruta_modelo_res)
        self.scaler = joblib.load(ruta_scaler_res)
        
        # Cargar motor de reglas
        self.motor_reglas = MotorReglas(ruta_reglas_res)
        
    def evaluar_paciente_completo(self, datos_paciente_dict, datos_paciente_df=None):
        # Si no se proporciona DataFrame, se construye a partir del diccionario
        if datos_paciente_df is None:
            datos_paciente_df = pd.DataFrame([datos_paciente_dict])

        # Asegurar orden consistente de columnas para el modelo de ML
        cols_presentes = [c for c in self.COLUMNAS_MODELO if c in datos_paciente_df.columns]
        df_ml = datos_paciente_df[cols_presentes] if len(cols_presentes) == len(self.COLUMNAS_MODELO) else datos_paciente_df

        # 1. Predicción del Modelo ML (ej. MLP, SVM, KNN)
        datos_escalados = self.scaler.transform(df_ml)
        pred_ml = self.modelo.predict(datos_escalados)[0]
        prob_ml = self.modelo.predict_proba(datos_escalados)[0][1] # Probabilidad de riesgo (clase 1)
        
        # 2. Evaluación del Motor de Reglas (JSON)
        diagnosticos_reglas = self.motor_reglas.evaluar_paciente(datos_paciente_dict)
        
        # 3. Fusión Híbrida de Decisiones
        alerta_experta = len(diagnosticos_reglas) > 0
        confianza_max_reglas = diagnosticos_reglas[0]['confianza'] if alerta_experta else 0.0
        
        # Lógica de combinación ponderada (60% ML, 40% Conocimiento Experto)
        score_hibrido = (prob_ml * 0.6) + (confianza_max_reglas * 0.4)
        decision_final = 1 if score_hibrido > 0.5 else 0
        
        return {
            "prediccion_ml": int(pred_ml),
            "probabilidad_ml": float(prob_ml),
            "reglas_activadas": len(diagnosticos_reglas),
            "confianza_reglas": float(confianza_max_reglas),
            "score_hibrido": float(score_hibrido),
            "decision_final_triaje": int(decision_final),
            "detalle_reglas": diagnosticos_reglas
        }

# --- EJEMPLO DE USO ---
if __name__ == "__main__":
    sistema = SistemaHibrido(
        ruta_modelo="../models/mejor_modelo.pkl",
        ruta_scaler="../models/scaler.pkl",
        ruta_reglas="../reglas_experto.json"
    )
    
    # Datos de prueba de un paciente
    paciente_dict = {
        "fiebre": 39.2, "edad": 62, "presion_sistolica": 150,
        "presion_diastolica": 95, "bmi": 31.0, "glucosa": 190, "tos_seca": 1
    }
    paciente_df = pd.DataFrame([paciente_dict])
    
    resultado = sistema.evaluar_paciente_completo(paciente_dict, paciente_df)
    print("🏥 Resultado del Sistema Híbrido:")
    for k, v in resultado.items():
        print(f"  • {k}: {v}")
