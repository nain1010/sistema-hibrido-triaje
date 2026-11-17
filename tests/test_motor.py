import unittest
from src.motor_reglas import MotorReglas

class TestMotorReglas(unittest.TestCase):
    def setUp(self):
        self.motor = MotorReglas("reglas_experto.json")

    def test_regla_r01_infeccion_respiratoria(self):
        paciente = {
            "fiebre": 39.0,
            "tos_seca": 1,
            "edad": 30,
            "presion_sistolica": 120,
            "presion_diastolica": 80,
            "bmi": 24.0,
            "glucosa": 90
        }
        res = self.motor.evaluar_paciente(paciente)
        ids = [r["regla_id"] for r in res]
        self.assertIn("R01", ids)

    def test_regla_r02_riesgo_metabolico(self):
        paciente = {
            "fiebre": 36.5,
            "tos_seca": 0,
            "edad": 65,
            "presion_sistolica": 120,
            "presion_diastolica": 80,
            "bmi": 24.0,
            "glucosa": 190
        }
        res = self.motor.evaluar_paciente(paciente)
        ids = [r["regla_id"] for r in res]
        self.assertIn("R02", ids)

    def test_regla_r03_hipertension(self):
        paciente = {
            "fiebre": 36.5,
            "tos_seca": 0,
            "edad": 40,
            "presion_sistolica": 145,
            "presion_diastolica": 85,
            "bmi": 24.0,
            "glucosa": 90
        }
        res = self.motor.evaluar_paciente(paciente)
        ids = [r["regla_id"] for r in res]
        self.assertIn("R03", ids)

    def test_regla_r04_riesgo_cardiovascular(self):
        paciente = {
            "fiebre": 36.5,
            "tos_seca": 0,
            "edad": 40,
            "presion_sistolica": 135,
            "presion_diastolica": 85,
            "bmi": 31.5,
            "glucosa": 90
        }
        res = self.motor.evaluar_paciente(paciente)
        ids = [r["regla_id"] for r in res]
        self.assertIn("R04", ids)

    def test_regla_r05_infeccion_leve(self):
        paciente = {
            "fiebre": 37.8,
            "tos_seca": 0,
            "edad": 25,
            "presion_sistolica": 120,
            "presion_diastolica": 80,
            "bmi": 22.0,
            "glucosa": 85
        }
        res = self.motor.evaluar_paciente(paciente)
        ids = [r["regla_id"] for r in res]
        self.assertIn("R05", ids)

    def test_regla_r06_precaucion_glucosa(self):
        paciente = {
            "fiebre": 36.5,
            "tos_seca": 0,
            "edad": 50,
            "presion_sistolica": 120,
            "presion_diastolica": 80,
            "bmi": 24.0,
            "glucosa": 150
        }
        res = self.motor.evaluar_paciente(paciente)
        ids = [r["regla_id"] for r in res]
        self.assertIn("R06", ids)

    def test_regla_r07_alerta_diastolica(self):
        paciente = {
            "fiebre": 36.5,
            "tos_seca": 0,
            "edad": 40,
            "presion_sistolica": 120,
            "presion_diastolica": 95,
            "bmi": 24.0,
            "glucosa": 90
        }
        res = self.motor.evaluar_paciente(paciente)
        ids = [r["regla_id"] for r in res]
        self.assertIn("R07", ids)

    def test_regla_r08_fiebre_origen_desconocido(self):
        paciente = {
            "fiebre": 38.5,
            "tos_seca": 0,
            "edad": 40,
            "presion_sistolica": 120,
            "presion_diastolica": 80,
            "bmi": 24.0,
            "glucosa": 90
        }
        res = self.motor.evaluar_paciente(paciente)
        ids = [r["regla_id"] for r in res]
        self.assertIn("R08", ids)

    def test_ordenamiento_confianza(self):
        # Paciente con R01 (0.92) y R03 (0.85)
        paciente = {
            "fiebre": 39.0,
            "tos_seca": 1,
            "edad": 45,
            "presion_sistolica": 145,
            "presion_diastolica": 80,
            "bmi": 24.0,
            "glucosa": 90
        }
        res = self.motor.evaluar_paciente(paciente)
        self.assertGreaterEqual(len(res), 2)
        confianzas = [r["confianza"] for r in res]
        self.assertEqual(confianzas, sorted(confianzas, reverse=True))

if __name__ == "__main__":
    unittest.main()
