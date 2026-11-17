#!/usr/bin/env python3
"""
Punto de Entrada Principal (Main) - Sistema Híbrido de Triaje Clínico
Permite ejecutar, probar y demostrar todos los componentes de los Sprints 1, 2 y 3.
"""

import sys
import os
import subprocess
import json

from src.evaluador_integral import EvaluadorTriajeIntegral
from src.motor_reglas import MotorReglas
from src.sistema_hibrido import SistemaHibrido
from src.entrenar_evaluar import entrenar_y_evaluar
from src.no_supervisado import ejecutar_aprendizaje_no_supervisado
from src.agente_qlearning import entrenar_agente_qlearning


PACIENTES_EJEMPLO = {
    "1": {
        "nombre": "Paciente Crítico (Fiebre alta, tos seca, comorbilidades)",
        "datos": {
            "fiebre": 39.2, "edad": 62, "presion_sistolica": 150,
            "presion_diastolica": 95, "bmi": 31.0, "glucosa": 190, "tos_seca": 1
        }
    },
    "2": {
        "nombre": "Paciente Estable / Sano (Signos vitales normales)",
        "datos": {
            "fiebre": 36.5, "edad": 28, "presion_sistolica": 115,
            "presion_diastolica": 75, "bmi": 22.0, "glucosa": 85, "tos_seca": 0
        }
    },
    "3": {
        "nombre": "Paciente con Hipertensión Severa",
        "datos": {
            "fiebre": 36.8, "edad": 55, "presion_sistolica": 165,
            "presion_diastolica": 102, "bmi": 27.5, "glucosa": 105, "tos_seca": 0
        }
    },
    "4": {
        "nombre": "Paciente con Sospecha Infección Respiratoria Leve",
        "datos": {
            "fiebre": 37.8, "edad": 24, "presion_sistolica": 118,
            "presion_diastolica": 78, "bmi": 23.0, "glucosa": 90, "tos_seca": 1
        }
    }
}


def imprimir_evaluacion_completa(resultado):
    print("\n" + "="*70)
    print("🏥 INFORME DE TRIAJE INTEGRAL (SPRINTS 1, 2 Y 3)")
    print("="*70)
    
    # Datos del paciente
    p = resultado["datos_ingresados"]
    print(f"\n📋 Signos Vitales Ingresados:")
    print(f"  • Temperatura: {p['fiebre']} °C | Edad: {p['edad']} años")
    print(f"  • Presión Arterial: {p['presion_sistolica']}/{p['presion_diastolica']} mmHg")
    print(f"  • IMC: {p['bmi']} | Glucosa: {p['glucosa']} mg/dL | Tos Seca: {'Sí' if p['tos_seca'] == 1 else 'No'}")

    # Sprint 1
    s1 = resultado["sprint_1_reglas"]
    print(f"\n1️⃣ [Sprint 1] Motor de Reglas Expertas (JSON):")
    print(f"  • Reglas activadas: {s1['reglas_activadas']} | Confianza máxima: {s1['confianza_maxima'] * 100:.1f}%")
    for r in s1["detalle_reglas"]:
        print(f"    - [{r['regla_id']}] {r['conclusion']} (Confianza: {r['confianza']}) -> {r['descripcion']}")

    # Sprint 2
    s2 = resultado["sprint_2_supervisado_hibrido"]
    print(f"\n2️⃣ [Sprint 2] Aprendizaje Supervisado y Fusión Híbrida:")
    print(f"  • Probabilidad de Riesgo (ML): {s2['probabilidad_riesgo_ml'] * 100:.2f}%")
    print(f"  • Score Híbrido Ponderado (60% ML + 40% Reglas): {s2['score_hibrido'] * 100:.2f}%")
    print(f"  • 🎯 DECISIÓN FINAL DE TRIAJE: {s2['decision_final_triaje']}")

    # Sprint 3 Parte A
    s3a = resultado["sprint_3_no_supervisado"]
    print(f"\n3️⃣ [Sprint 3-A] Aprendizaje No Supervisado (K-Means y SOM):")
    print(f"  • Clúster K-Means Asignado: Clúster {s3a['cluster_kmeans']} ({s3a['cluster_interpretacion']})")
    print(f"  • Coordenada Neuronal en Mapa SOM (7x7): {s3a['coordenadas_som']}")

    # Sprint 3 Parte B
    s3b = resultado["sprint_3_refuerzo"]
    print(f"\n4️⃣ [Sprint 3-B] Aprendizaje por Refuerzo (Q-Learning):")
    print(f"  • Estado Clínico en MDP: {s3b['estado_mdp']}")
    print(f"  • 🚦 ACCIÓN ÓPTIMA APRENDIDA (π*): {s3b['accion_optima_qlearning'].upper()}")
    print(f"  • Valores Q del Estado: {s3b['valores_q_estado']}")
    print("="*70 + "\n")


def menu():
    raiz = os.path.dirname(os.path.abspath(__file__))
    evaluador = EvaluadorTriajeIntegral(raiz)

    while True:
        print("\n" + "="*50)
        print("🏥 SISTEMA HÍBRIDO DE TRIAJE CLÍNICO")
        print("   Integrador: Sprints 1, 2 y 3")
        print("="*50)
        print("1. Evaluar Paciente Predefinido (Casos de Prueba)")
        print("2. Evaluar Paciente con Datos Manuales")
        print("3. Demostración Sprint 1: Motor de Reglas (JSON)")
        print("4. Demostración Sprint 2: Modelos ML y Fusión Híbrida")
        print("5. Demostración Sprint 3: Clustering K-Means y Mapa SOM")
        print("6. Demostración Sprint 3: Agente de Refuerzo Q-Learning")
        print("7. Lanzar Dashboard Web Interactivo (FastAPI)")
        print("8. Ejecutar Batería de Pruebas Unitarias Automatizadas")
        print("9. Salir")
        
        opcion = input("\nSeleccione una opción (1-9): ").strip()

        if opcion == "1":
            print("\nCasos de prueba disponibles:")
            for k, v in PACIENTES_EJEMPLO.items():
                print(f"  {k}. {v['nombre']}")
            sel = input("Seleccione caso (1-4): ").strip()
            if sel in PACIENTES_EJEMPLO:
                res = evaluador.evaluar_paciente(PACIENTES_EJEMPLO[sel]["datos"])
                imprimir_evaluacion_completa(res)
            else:
                print("Opción inválida.")

        elif opcion == "2":
            print("\nIngrese los datos del paciente:")
            try:
                fiebre = float(input("Temperatura corporal (°C) [ej: 38.5]: "))
                edad = int(input("Edad (años) [ej: 50]: "))
                ps = float(input("Presión sistólica (mmHg) [ej: 130]: "))
                pd_val = float(input("Presión diastólica (mmHg) [ej: 85]: "))
                bmi = float(input("Índice de Masa Corporal (BMI) [ej: 28.5]: "))
                glucosa = float(input("Glucosa (mg/dL) [ej: 120]: "))
                tos = int(input("Tos seca (1: Sí, 0: No): "))

                paciente = {
                    "fiebre": fiebre, "edad": edad, "presion_sistolica": ps,
                    "presion_diastolica": pd_val, "bmi": bmi, "glucosa": glucosa, "tos_seca": tos
                }
                res = evaluador.evaluar_paciente(paciente)
                imprimir_evaluacion_completa(res)
            except ValueError:
                print("❌ Error: Ingrese valores numéricos válidos.")

        elif opcion == "3":
            print("\n--- Ejecutando Motor de Reglas (Sprint 1) ---")
            p = PACIENTES_EJEMPLO["1"]["datos"]
            motor = MotorReglas("reglas_experto.json")
            diags = motor.evaluar_paciente(p)
            print(f"Paciente evaluado: Fiebre {p['fiebre']}°C, Tos seca: {p['tos_seca']}")
            for d in diags:
                print(f" • [{d['regla_id']}] {d['conclusion']} ({d['confianza']*100:.0f}%) -> {d['descripcion']}")

        elif opcion == "4":
            print("\n--- Re-evaluando Modelos Supervisados y Fusión (Sprint 2) ---")
            entrenar_y_evaluar(raiz)

        elif opcion == "5":
            print("\n--- Ejecutando K-Means y SOM (Sprint 3 Parte A) ---")
            ejecutar_aprendizaje_no_supervisado(raiz)

        elif opcion == "6":
            print("\n--- Entrenando y Evaluando Agente Q-Learning (Sprint 3 Parte B) ---")
            entrenar_agente_qlearning(raiz, num_episodios=3000)

        elif opcion == "7":
            print("\n🌐 Iniciando Dashboard Web en http://localhost:8000 ...")
            print("Presiona Ctrl+C en la terminal para detener el servidor.")
            subprocess.run([sys.executable, "-m", "uvicorn", "src.app_web:app", "--host", "0.0.0.0", "--port", "8000"])

        elif opcion == "8":
            print("\n🧪 Ejecutando todas las pruebas unitarias...")
            subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"])

        elif opcion == "9":
            print("\n¡Gracias por utilizar el Sistema Híbrido de Triaje Clínico!")
            break
        else:
            print("Opción no reconocida. Intente de nuevo.")


if __name__ == "__main__":
    menu()
