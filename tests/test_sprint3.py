import unittest
import os
import joblib
import numpy as np
import pandas as pd

class TestSprint3(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kmeans = joblib.load("models/kmeans.pkl")
        cls.som = joblib.load("models/som.pkl")
        cls.scaler = joblib.load("models/scaler.pkl")
        cls.q_table = joblib.load("models/q_table.pkl")
        cls.columnas = ["fiebre", "edad", "presion_sistolica", "presion_diastolica", "bmi", "glucosa", "tos_seca"]

    def test_kmeans_clusterizacion(self):
        paciente_df = pd.DataFrame([[39.5, 65, 160, 95, 33.0, 185.0, 1]], columns=self.columnas)
        paciente_scaled = self.scaler.transform(paciente_df)
        cluster = self.kmeans.predict(paciente_scaled)
        self.assertIn(cluster[0], [0, 1, 2])

    def test_som_proyeccion(self):
        paciente_df = pd.DataFrame([[36.5, 25, 115, 75, 22.0, 85.0, 0]], columns=self.columnas)
        paciente_scaled = self.scaler.transform(paciente_df)
        w = self.som.winner(paciente_scaled[0])
        self.assertEqual(len(w), 2)
        self.assertTrue(0 <= w[0] < 7 and 0 <= w[1] < 7)

    def test_qlearning_politica_optima(self):
        # Estados: 0: Estable, 1: Precaución, 2: Crítico
        # Acciones: 0: Alta, 1: Observación, 2: Urgencia
        self.assertEqual(np.argmax(self.q_table[0]), 0) # Estable -> Alta a casa
        self.assertEqual(np.argmax(self.q_table[1]), 1) # Precaución -> Observación
        self.assertEqual(np.argmax(self.q_table[2]), 2) # Crítico -> Urgencia inmediata
        
        # Penalización severa a falso negativo en paciente crítico (Q[2, 0] ~ -50.0)
        self.assertAlmostEqual(self.q_table[2, 0], -50.0, places=1)

if __name__ == "__main__":
    unittest.main()
