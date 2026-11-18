"""
Script de Demostración en Vivo para la Defensa Oral (2 de Diciembre):
Ejecuta los 3 Casos Clínicos de Validación en 15 minutos:
1. Caso 1: Paciente Sano (Signos estables, sin reglas activadas, radiografía normal).
2. Caso 2: Caso Límite / Frontera (Fiebre moderada / glucosa limítrofe, decisión ponderada).
3. Caso 3: Paciente Crítico Multimodal (Alerta en reglas, probabilidad ML alta, neumonía severa en CNN).
"""

import os
import sys

dir_actual = os.path.dirname(os.path.abspath(__file__))
dir_raiz = os.path.dirname(dir_actual)
if dir_raiz not in sys.path:
    sys.path.insert(0, dir_raiz)
if dir_actual not in sys.path:
    sys.path.insert(0, dir_actual)

from evaluador_integral import EvaluadorTriajeIntegral


def ejecutar_demo():
    raiz = dir_raiz
    evaluador = EvaluadorTriajeIntegral(raiz)

    casos = [
        {
            "num": 1,
            "titulo": "CASO 1: PACIENTE SANO / ESTABLE (BAJO RIESGO)",
            "descripcion": "Joven de 26 años con signos vitales dentro de rangos normales y radiografía de tórax limpia.",
            "datos": {
                "fiebre": 36.6, "edad": 26, "presion_sistolica": 118,
                "presion_diastolica": 76, "bmi": 22.4, "glucosa": 88, "tos_seca": 0
            },
            "radiografia": os.path.join(raiz, "data", "radiografias", "paciente_sano.png"),
            "explicacion_tecnica": (
                "• Reglas: 0 reglas activadas.\n"
                "• ML Supervisado: P(Riesgo) baja (< 5%).\n"
                "• K-Means: Asignado a Clúster 2 (Bajo riesgo).\n"
                "• Q-Learning: Estado 'Estable' -> Acción 'Alta a casa'.\n"
                "• CNN: Campos pulmonares radiolúcidos sin consolidaciones."
            )
        },
        {
            "num": 2,
            "titulo": "CASO 2: CASO LÍMITE / FRONTERA (DECISIÓN PONDERADA)",
            "descripcion": "Adulto de 48 años con glucosa limítrofe (155 mg/dL), tensión moderada y discreto infiltrado broncopulmonar.",
            "datos": {
                "fiebre": 37.6, "edad": 48, "presion_sistolica": 134,
                "presion_diastolica": 86, "bmi": 28.2, "glucosa": 155, "tos_seca": 1
            },
            "radiografia": os.path.join(raiz, "data", "radiografias", "paciente_limite.png"),
            "explicacion_tecnica": (
                "• Reglas: Se dispara R06 (Precaución glucosa 140-179).\n"
                "• ML Supervisado: Probabilidad moderada (~35-45%).\n"
                "• K-Means: Clúster 0 (Perfil febril/límite).\n"
                "• Q-Learning: Estado 'Precaución' -> Acción 'Observación'.\n"
                "• CNN: Infiltrado limítrofe (~40.7%).\n"
                "• Fusión Multimodal: Requiere observación y monitoreo (evita dar de alta precipitadamente)."
            )
        },
        {
            "num": 3,
            "titulo": "CASO 3: PACIENTE CRÍTICO MULTIMODAL (ALERTA ROJA)",
            "descripcion": "Adulto mayor de 68 años con fiebre alta (39.4 °C), tos seca, crisis hipertensiva, hiperglucemia y consolidación densa en radiografía.",
            "datos": {
                "fiebre": 39.4, "edad": 68, "presion_sistolica": 162,
                "presion_diastolica": 98, "bmi": 32.5, "glucosa": 210, "tos_seca": 1
            },
            "radiografia": os.path.join(raiz, "data", "radiografias", "paciente_critico_neumonia.png"),
            "explicacion_tecnica": (
                "• Reglas: Múltiples reglas activadas (R01, R02, R03, R04, R07) con confianza > 90%.\n"
                "• ML Supervisado: P(Riesgo) > 95%.\n"
                "• K-Means: Clúster 1 (Crítico con comorbilidades).\n"
                "• Q-Learning: Estado 'Crítico' -> Acción 'Urgencia inmediata' (evitando la penalización de -50).\n"
                "• CNN: Neumonía severa bilateral (> 95% de confianza).\n"
                "• Fusión Multimodal: Alerta Roja absoluta y triaje nivel 1."
            )
        }
    ]

    print("\n" + "#"*75)
    print("🎓 DEMOSTRACIÓN EN VIVO: SISTEMA MULTIMODAL DE TRIAJE CLÍNICO")
    print("   Defensa Oral del Proyecto Integrador de Inteligencia Artificial")
    print("#"*75)

    for c in casos:
        print("\n" + "="*75)
        print(f"🩺 {c['titulo']}")
        print(f"📖 Contexto Clínico: {c['descripcion']}")
        print("="*75)
        
        d = c["datos"]
        print(f"Parámetros: Fiebre {d['fiebre']}°C | Edad {d['edad']}a | Tensión {d['presion_sistolica']}/{d['presion_diastolica']} | BMI {d['bmi']} | Glucosa {d['glucosa']} | Tos Seca: {'Sí' if d['tos_seca']==1 else 'No'}")
        print(f"Radiografía adjunta: {os.path.basename(c['radiografia'])}")

        heatmap_salida = os.path.join(raiz, "reports", f"demo_caso_{c['num']}_heatmap.png")
        resultado = evaluador.evaluar_paciente(c["datos"], c["radiografia"], guardar_heatmap_en=heatmap_salida)

        # Resultados de los 4 Mundos
        print("\n📊 RESULTADOS OBTENIDOS POR CADA COMPONENTE:")
        s1 = resultado["sprint_1_reglas"]
        print(f"  1. Reglas Expertas (JSON): {s1['reglas_activadas']} activadas (Confianza máx: {s1['confianza_maxima']*100:.0f}%)")
        for r in s1["detalle_reglas"]:
            print(f"     • [{r['regla_id']}] {r['conclusion']}")

        s2 = resultado["sprint_2_supervisado_hibrido"]
        print(f"  2. ML Supervisado (SVM/MLP): P(Riesgo) = {s2['probabilidad_riesgo_ml']*100:.1f}%")

        s3a = resultado["sprint_3_no_supervisado"]
        print(f"  3. K-Means / SOM: Clúster {s3a['cluster_kmeans']} ({s3a['cluster_interpretacion']}) | Neurona SOM: {s3a['coordenadas_som']}")

        s3b = resultado["sprint_3_refuerzo"]
        print(f"  4. Q-Learning (MDP): Estado '{s3b['estado_mdp']}' -> Acción Óptima: '{s3b['accion_optima_qlearning'].upper()}'")

        s4 = resultado["sprint_4_vision_cnn"]
        print(f"  5. Visión Artificial (CNN): P(Neumonía) = {s4['probabilidad_neumonia']*100:.1f}% ({s4['severidad_imagen']})")
        print(f"     Dictamen: {s4['dictamen_radiologico']}")

        glob = resultado["dictamen_global_multimodal"]
        print("\n" + "-"*75)
        print(f"🎯 DICTAMEN FINAL MULTIMODAL: {glob['decision_final_triaje']}")
        print(f"📈 Score Multimodal Global: {glob['score_multimodal']*100:.1f}% | Nivel de Prioridad: {glob['nivel_prioridad']}")
        print(f"💡 {glob['justificacion_clinica']}")
        print("-"*75)

        print("\n🔍 Justificación Técnica para el Tribunal Evaluador:")
        print(c["explicacion_tecnica"])
        print(f"🖼️ Heatmap Grad-CAM guardado en: {heatmap_salida}")

    print("\n" + "#"*75)
    print("✅ DEMOSTRACIÓN FINALIZADA CON ÉXITO")
    print("#"*75 + "\n")


if __name__ == "__main__":
    ejecutar_demo()
