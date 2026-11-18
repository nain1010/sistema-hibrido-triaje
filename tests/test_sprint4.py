import unittest
import os
import numpy as np
from src.cnn_radiografias import ClasificadorCNNRadiografias
from src.evaluador_integral import EvaluadorTriajeIntegral

class TestSprint4(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cls.clasificador_cnn = ClasificadorCNNRadiografias()
        cls.evaluador = EvaluadorTriajeIntegral(cls.raiz)
        cls.rx_sano = os.path.join(cls.raiz, "data", "radiografias", "paciente_sano.png")
        cls.rx_critico = os.path.join(cls.raiz, "data", "radiografias", "paciente_critico_neumonia.png")
        cls.rx_limite = os.path.join(cls.raiz, "data", "radiografias", "paciente_limite.png")

    def test_cnn_paciente_sano(self):
        res = self.clasificador_cnn.analizar_radiografia(self.rx_sano)
        self.assertLess(res["probabilidad_neumonia"], 0.35)
        self.assertEqual(res["severidad_imagen"], "BAJA")

    def test_cnn_paciente_critico(self):
        res = self.clasificador_cnn.analizar_radiografia(self.rx_critico)
        self.assertGreater(res["probabilidad_neumonia"], 0.70)
        self.assertEqual(res["severidad_imagen"], "ALTA")

    def test_evaluacion_multimodal_integrada(self):
        paciente_critico = {
            "fiebre": 39.4, "edad": 68, "presion_sistolica": 162,
            "presion_diastolica": 98, "bmi": 32.5, "glucosa": 210, "tos_seca": 1
        }
        res = self.evaluador.evaluar_paciente(paciente_critico, self.rx_critico)
        self.assertIn("sprint_4_vision_cnn", res)
        self.assertIsNotNone(res["sprint_4_vision_cnn"])
        self.assertEqual(res["dictamen_global_multimodal"]["nivel_prioridad"], 1)

if __name__ == "__main__":
    unittest.main()
