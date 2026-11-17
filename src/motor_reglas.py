import json
import os

class MotorReglas:
    def __init__(self, ruta_json):
        # Resolver ruta: intentar ruta dada, luego relativa al script o a la raíz
        ruta_resuelta = ruta_json
        if not os.path.exists(ruta_resuelta):
            dir_script = os.path.dirname(os.path.abspath(__file__))
            candidato_1 = os.path.normpath(os.path.join(dir_script, ruta_json))
            candidato_2 = os.path.normpath(os.path.join(dir_script, "..", os.path.basename(ruta_json)))
            candidato_3 = os.path.normpath(os.path.join(os.getcwd(), os.path.basename(ruta_json)))
            
            if os.path.exists(candidato_1):
                ruta_resuelta = candidato_1
            elif os.path.exists(candidato_2):
                ruta_resuelta = candidato_2
            elif os.path.exists(candidato_3):
                ruta_resuelta = candidato_3
            else:
                raise FileNotFoundError(f"No se encontró el archivo de reglas en: {ruta_json}")
        
        with open(ruta_resuelta, 'r', encoding='utf-8') as f:
            self.base_conocimiento = json.load(f)
            
    def evaluar_paciente(self, paciente):
        diagnosticos = []
        
        for regla in self.base_conocimiento.get('reglas', []):
            cumple_todas = True
            
            for cond in regla.get('condiciones', []):
                var = cond['variable']
                op = cond['operador']
                val_ref = cond.get('valor', cond.get('vlr'))
                
                # Obtener el valor del paciente
                val_paciente = paciente.get(var)
                if val_paciente is None:
                    cumple_todas = False
                    break
                
                # Evaluar operador lógico
                if op == '>' and not (val_paciente > val_ref):
                    cumple_todas = False
                    break
                elif op == '>=' and not (val_paciente >= val_ref):
                    cumple_todas = False
                    break
                elif op == '<' and not (val_paciente < val_ref):
                    cumple_todas = False
                    break
                elif op == '<=' and not (val_paciente <= val_ref):
                    cumple_todas = False
                    break
                elif op == '==' and not (val_paciente == val_ref):
                    cumple_todas = False
                    break
                elif op == '!=' and not (val_paciente != val_ref):
                    cumple_todas = False
                    break
            
            if cumple_todas:
                diagnosticos.append({
                    "regla_id": regla["id"],
                    "conclusion": regla["conclusion"],
                    "confianza": regla["confianza"],
                    "descripcion": regla["descripcion"]
                })
                
        # Ordenar de mayor a menor confianza
        diagnosticos = sorted(diagnosticos, key=lambda x: x['confianza'], reverse=True)
        return diagnosticos

# --- PRUEBA DEL MOTOR ---
if __name__ == "__main__":
    motor = MotorReglas("../reglas_experto.json")
    
    # Paciente de prueba con fiebre alta y tos seca
    paciente_prueba = {
        "fiebre": 39.0,
        "edad": 45,
        "presion_sistolica": 120,
        "presion_diastolica": 80,
        "bmi": 25.0,
        "glucosa": 95,
        "tos_seca": 1
    }
    
    resultados = motor.evaluar_paciente(paciente_prueba)
    print("🔍 Diagnósticos generados por el Motor de Reglas:")
    for r in resultados:
        print(f"  • [{r['regla_id']}] {r['conclusion']} (Confianza: {r['confianza']}) -> {r['descripcion']}")
