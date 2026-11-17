"""
Módulo de Aprendizaje por Refuerzo para Triaje Clínico (Sprint 3 - Parte B):
- Definición formal del MDP (Estados, Acciones, Función de Recompensa Penalizando Falsos Negativos).
- Implementación del algoritmo Q-Learning (Ecuación de Bellman, política e-greedy).
- Entrenamiento y convergencia hacia la política óptima de triaje.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class EntornoTriajeMDP:
    """
    Entorno de Triaje modelado como Proceso de Decisión de Markov (MDP):
    - Estados (S):
      0: Estable (Paciente sin signos críticos ni factores de alarma)
      1: Precaución (Paciente con signos limítrofes / riesgo moderado)
      2: Crítico (Paciente con fiebre severa, crisis hipertensiva o hiperglucemia)
    
    - Acciones (A):
      0: Dar de alta a casa (Atención ambulatoria / autocuidado)
      1: Enviar a observación (Monitoreo continuo en sala)
      2: Urgencia inmediata (Atención médica prioritaria)
      
    - Matriz de Recompensas R(s, a):
      Especialmente diseñada para penalizar de forma crítica los Falsos Negativos
      (dar de alta a un paciente grave: penalización de -50).
    """

    NOMBRES_ESTADOS = ["Estable", "Precaución", "Crítico"]
    NOMBRES_ACCIONES = ["Alta a casa", "Observación", "Urgencia inmediata"]

    def __init__(self, ruta_datos=None):
        # Matriz de recompensa: filas = estados, columnas = acciones
        self.recompensas = np.array([
            # Alta(0)   Obs(1)   Urgencia(2)
            [ +10.0,    -2.0,     -15.0 ],  # Estado 0: Estable
            [ -20.0,   +15.0,      -5.0 ],  # Estado 1: Precaución
            [ -50.0,   -10.0,     +20.0 ]   # Estado 2: Crítico (FN penalizado severamente con -50)
        ])

        self.num_estados = 3
        self.num_acciones = 3
        self.pacientes_estados = self._cargar_estados_pacientes(ruta_datos)
        self.indice_actual = 0
        self.estado_actual = 0
        self.reset()

    def _clasificar_estado_clinico(self, row):
        # Criterios clínicos basados en conocimiento experto
        fiebre = row.get("fiebre", 36.5)
        tos = row.get("tos_seca", 0)
        presion_s = row.get("presion_sistolica", 120)
        presion_d = row.get("presion_diastolica", 80)
        glucosa = row.get("glucosa", 90)
        edad = row.get("edad", 30)

        # Criterio Crítico (Estado 2)
        if (fiebre > 38.5 and tos == 1) or (glucosa > 180 and edad >= 60) or (presion_s > 140) or (presion_d > 90):
            return 2
        # Criterio Precaución (Estado 1)
        elif (fiebre > 37.5) or (presion_s > 130) or (glucosa >= 140):
            return 1
        # Criterio Estable (Estado 0)
        else:
            return 0

    def _cargar_estados_pacientes(self, ruta_datos):
        if ruta_datos and os.path.exists(ruta_datos):
            df = pd.read_csv(ruta_datos)
            return [self._clasificar_estado_clinico(r) for _, r in df.iterrows()]
        # Distribución de simulación estándar en caso de no proveer archivo
        return np.random.choice([0, 1, 2], size=300, p=[0.50, 0.30, 0.20]).tolist()

    def reset(self):
        if len(self.pacientes_estados) > 0:
            self.estado_actual = np.random.choice(self.pacientes_estados)
        else:
            self.estado_actual = np.random.choice([0, 1, 2], p=[0.50, 0.30, 0.20])
        return self.estado_actual

    def step(self, accion):
        recompensa = self.recompensas[self.estado_actual, accion]
        
        # En triaje por turnos, un episodio representa la evaluación de un paciente
        # y la llegada del siguiente paciente
        nuevo_estado = self.reset()
        done = True # Cada paciente constituye una decisión de triaje
        
        return nuevo_estado, recompensa, done, {}


class AgenteQLearning:
    def __init__(self, num_estados=3, num_acciones=3, lr=0.1, gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.num_estados = num_estados
        self.num_acciones = num_acciones
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Tabla Q inicializada en ceros
        self.q_table = np.zeros((num_estados, num_acciones))

    def elegir_accion(self, estado):
        # Política épsilon-greedy
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_acciones)
        return int(np.argmax(self.q_table[estado]))

    def actualizar_q(self, estado, accion, recompensa, nuevo_estado, done):
        mejor_siguiente_q = np.max(self.q_table[nuevo_estado]) if not done else 0.0
        td_target = recompensa + (self.gamma * mejor_siguiente_q)
        td_error = td_target - self.q_table[estado, accion]
        self.q_table[estado, accion] += self.lr * td_error

    def decaer_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def entrenar_agente_qlearning(raiz_proyecto, num_episodios=3000):
    ruta_models = os.path.join(raiz_proyecto, "models")
    ruta_reports = os.path.join(raiz_proyecto, "reports")
    ruta_datos = os.path.join(raiz_proyecto, "data", "processed", "dataset_clean.csv")
    os.makedirs(ruta_models, exist_ok=True)
    os.makedirs(ruta_reports, exist_ok=True)

    entorno = EntornoTriajeMDP(ruta_datos)
    agente = AgenteQLearning(
        num_estados=entorno.num_estados,
        num_acciones=entorno.num_acciones,
        lr=0.1,
        gamma=0.90,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.998
    )

    recompensas_historial = []

    print("\n" + "="*60)
    print("INICIANDO ENTRENAMIENTO DEL AGENTE Q-LEARNING")
    print(f"Episodios: {num_episodios} | Gamma: {agente.gamma} | Alpha: {agente.lr}")
    print("="*60)

    for episodio in range(1, num_episodios + 1):
        estado = entorno.reset()
        accion = agente.elegir_accion(estado)
        nuevo_estado, recompensa, done, _ = entorno.step(accion)
        
        agente.actualizar_q(estado, accion, recompensa, nuevo_estado, done)
        agente.decaer_epsilon()
        recompensas_historial.append(recompensa)

    # Guardar Q-table
    ruta_qtable = os.path.join(ruta_models, "q_table.pkl")
    joblib.dump(agente.q_table, ruta_qtable)
    print(f"\nQ-Table guardada en: {ruta_qtable}")

    # Visualizar Q-table resultante
    df_q = pd.DataFrame(
        agente.q_table,
        index=entorno.NOMBRES_ESTADOS,
        columns=entorno.NOMBRES_ACCIONES
    )
    print("\n--- Tabla Q Convergida ---")
    print(df_q.round(2).to_string())

    # Política Óptima aprendida pi*(s) = argmax Q(s, a)
    politica_optima = {}
    print("\n--- POLÍTICA ÓPTIMA DE TRIAJE APRENDIDA (\u03c0*) ---")
    for s_idx, s_nombre in enumerate(entorno.NOMBRES_ESTADOS):
        mejor_a = int(np.argmax(agente.q_table[s_idx]))
        politica_optima[s_nombre] = entorno.NOMBRES_ACCIONES[mejor_a]
        print(f"  • Si el paciente está [{s_nombre:10s}] -> Decisión óptima: {entorno.NOMBRES_ACCIONES[mejor_a].upper()}")

    # Graficar convergencia
    ventana = 50
    recompensas_suavizadas = pd.Series(recompensas_historial).rolling(window=ventana).mean()

    plt.figure(figsize=(10, 5))
    plt.plot(recompensas_historial, alpha=0.15, color='gray', label='Recompensa por Episodio')
    plt.plot(recompensas_suavizadas, color='royalblue', linewidth=2, label=f'Media Móvil ({ventana} ep.)')
    plt.axhline(y=10, color='forestgreen', linestyle='--', alpha=0.7, label='Nivel Óptimo Teórico')
    plt.title("Convergencia del Aprendizaje por Refuerzo (Q-Learning) en Triaje", fontsize=12)
    plt.xlabel("Episodios de Entrenamiento")
    plt.ylabel("Recompensa Obtenida")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    ruta_grafico = os.path.join(ruta_reports, "qlearning_convergencia.png")
    plt.savefig(ruta_grafico, dpi=300)
    plt.close()
    print(f"Gráfico de convergencia guardado en: {ruta_grafico}")

    return agente.q_table, politica_optima


if __name__ == "__main__":
    raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    entrenar_agente_qlearning(raiz)
