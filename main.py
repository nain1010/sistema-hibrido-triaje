#!/usr/bin/env python3
"""
Punto de Entrada Principal (Main) - Sistema Multimodal de Triaje Clínico
Permite ejecutar, probar y demostrar todos los componentes de los Sprints 1, 2, 3 y 4.
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
from src.cnn_radiografias import ClasificadorCNNRadiografias
from src.demo_defensa import ejecutar_demo


def menu():
    raiz = os.path.dirname(os.path.abspath(__file__))
    evaluador = EvaluadorTriajeIntegral(raiz)
    clasificador_cnn = ClasificadorCNNRadiografias()

    while True:
        print("\n" + "="*60)
        print("🏥 SISTEMA HÍBRIDO Y MULTIMODAL DE TRIAJE CLÍNICO")
        print("   Integrador Completo: Sprints 1, 2, 3 y 4 (Cierre)")
        print("="*60)
        print("1. 🎓 Ejecutar Demostración en Vivo (3 Casos para Defensa Oral)")
        print("2. 🩺 Evaluar Paciente Multimodal (Signos Vitales + Radiografía)")
        print("3. 🩻 Analizar Imagen Radiográfica con CNN (Sprint 4)")
        print("4. 📋 Demostración Sprint 1: Motor de Reglas (JSON)")
        print("5. 🤖 Demostración Sprint 2: Modelos ML y Fusión Híbrida")
        print("6. 🔍 Demostración Sprint 3: Clustering K-Means y Mapa SOM")
        print("7. 🎮 Demostración Sprint 3: Agente de Refuerzo Q-Learning")
        print("8. 🌐 Lanzar Dashboard Web Interactivo (FastAPI)")
        print("9. 🧪 Ejecutar Suite de Pruebas Unitarias Automatizadas")
        print("10. Salir")
        
        opcion = input("\nSeleccione una opción (1-10): ").strip()

        if opcion == "1":
            ejecutar_demo()

        elif opcion == "2":
            print("\n--- Evaluación Multimodal de Paciente ---")
            try:
                fiebre = float(input("Temperatura corporal (°C) [ej: 38.8]: "))
                edad = int(input("Edad (años) [ej: 60]: "))
                ps = float(input("Presión sistólica (mmHg) [ej: 145]: "))
                pd_val = float(input("Presión diastólica (mmHg) [ej: 92]: "))
                bmi = float(input("Índice de Masa Corporal (BMI) [ej: 30.5]: "))
                glucosa = float(input("Glucosa (mg/dL) [ej: 185]: "))
                tos = int(input("Tos seca (1: Sí, 0: No): "))

                paciente = {
                    "fiebre": fiebre, "edad": edad, "presion_sistolica": ps,
                    "presion_diastolica": pd_val, "bmi": bmi, "glucosa": glucosa, "tos_seca": tos
                }

                print("\nOpciones de radiografía de tórax:")
                print(" 1. data/radiografias/paciente_sano.png")
                print(" 2. data/radiografias/paciente_limite.png")
                print(" 3. data/radiografias/paciente_critico_neumonia.png")
                print(" 4. Ninguna (solo tabular)")
                sel_rx = input("Seleccione radiografía (1-4): ").strip()

                mapa_rx = {
                    "1": os.path.join(raiz, "data", "radiografias", "paciente_sano.png"),
                    "2": os.path.join(raiz, "data", "radiografias", "paciente_limite.png"),
                    "3": os.path.join(raiz, "data", "radiografias", "paciente_critico_neumonia.png")
                }
                ruta_rx = mapa_rx.get(sel_rx, None)

                res = evaluador.evaluar_paciente(paciente, ruta_rx)
                glob = res["dictamen_global_multimodal"]

                print("\n" + "="*60)
                print("📋 RESULTADO DEL TRIAJE INTEGRAL MULTIMODAL")
                print("="*60)
                print(f"🎯 Dictamen: {glob['decision_final_triaje']}")
                print(f"📈 Score Multimodal: {glob['score_multimodal']*100:.1f}% | Nivel: {glob['nivel_prioridad']}")
                print(f"💡 Justificación: {glob['justificacion_clinica']}")
                if res["sprint_4_vision_cnn"]:
                    print(f"🩻 CNN Tórax: {res['sprint_4_vision_cnn']['dictamen_radiologico']} (P={res['sprint_4_vision_cnn']['probabilidad_neumonia']*100:.1f}%)")
                print("="*60)
            except ValueError:
                print("❌ Error: Ingrese valores numéricos válidos.")

        elif opcion == "3":
            print("\n--- Análisis de Radiografía con CNN ---")
            print("1. Paciente Sano")
            print("2. Paciente Límite")
            print("3. Paciente Crítico (Neumonía)")
            sel = input("Seleccione imagen (1-3): ").strip()
            mapa = {
                "1": "data/radiografias/paciente_sano.png",
                "2": "data/radiografias/paciente_limite.png",
                "3": "data/radiografias/paciente_critico_neumonia.png"
            }
            if sel in mapa:
                ruta = os.path.join(raiz, mapa[sel])
                res = clasificador_cnn.analizar_radiografia(ruta)
                print(f"\n[+] Radiografía analizada: {mapa[sel]}")
                print(f"  • Probabilidad Neumonía: {res['probabilidad_neumonia']*100:.1f}%")
                print(f"  • Severidad: {res['severidad_imagen']}")
                print(f"  • Dictamen: {res['dictamen_radiologico']}")
                print(f"  • Recomendación: {res['recomendacion_clinica']}")

        elif opcion == "4":
            print("\n--- Ejecutando Motor de Reglas (Sprint 1) ---")
            motor = MotorReglas("reglas_experto.json")
            p = {"fiebre": 39.0, "edad": 45, "presion_sistolica": 145, "presion_diastolica": 95, "bmi": 31.0, "glucosa": 190, "tos_seca": 1}
            diags = motor.evaluar_paciente(p)
            for d in diags:
                print(f" • [{d['regla_id']}] {d['conclusion']} ({d['confianza']*100:.0f}%) -> {d['descripcion']}")

        elif opcion == "5":
            print("\n--- Re-evaluando Modelos Supervisados y Fusión (Sprint 2) ---")
            entrenar_y_evaluar(raiz)

        elif opcion == "6":
            print("\n--- Ejecutando K-Means y SOM (Sprint 3 Parte A) ---")
            ejecutar_aprendizaje_no_supervisado(raiz)

        elif opcion == "7":
            print("\n--- Entrenando y Evaluando Agente Q-Learning (Sprint 3 Parte B) ---")
            entrenar_agente_qlearning(raiz, num_episodios=3000)

        elif opcion == "8":
            print("\n🌐 Iniciando Dashboard Web en http://localhost:8000 ...")
            print("Presiona Ctrl+C en la terminal para detener el servidor.")
            subprocess.run([sys.executable, "-m", "uvicorn", "src.app_web:app", "--host", "0.0.0.0", "--port", "8000"])

        elif opcion == "9":
            print("\n🧪 Ejecutando todas las pruebas unitarias...")
            subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"])

        elif opcion == "10":
            print("\n¡Mucho éxito en la defensa del 2 de Diciembre!")
            break


if __name__ == "__main__":
    menu()
