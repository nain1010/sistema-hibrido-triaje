import unittest
import os
import pandas as pd
from src.sistema_hibrido import SistemaHibrido

class TestSistemaHibrido(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sistema = SistemaHibrido(
            ruta_modelo="models/mejor_modelo.pkl",
            ruta_scaler="models/scaler.pkl",
            ruta_reglas="reglas_experto.json"
        )

    def test_paciente_alto_riesgo(self):
        paciente_dict = {
            "fiebre": 39.2, "edad": 62, "presion_sistolica": 150,
            "presion_diastolica": 95, "bmi": 31.0, "glucosa": 190, "tos_seca": 1
        }
        res = self.sistema.evaluar_paciente_completo(paciente_dict)
        self.assertEqual(res["decision_final_triaje"], 1)
        self.assertGreater(res["probabilidad_ml"], 0.5)
        self.assertGreater(res["reglas_activadas"], 0)
        self.assertGreater(res["score_hibrido"], 0.5)

    def test_paciente_sano_bajo_riesgo(self):
        paciente_dict = {
            "fiebre": 36.5, "edad": 28, "presion_sistolica": 115,
            "presion_diastolica": 75, "bmi": 22.0, "glucosa": 85, "tos_seca": 0
        }
        res = self.sistema.evaluar_paciente_completo(paciente_dict)
        self.assertEqual(res["decision_final_triaje"], 0)
        self.assertLess(res["probabilidad_ml"], 0.5)
        self.assertLess(res["score_hibrido"], 0.5)

if __name__ == "__main__":
    unittest.main()
