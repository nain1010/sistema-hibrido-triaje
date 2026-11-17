"""
Servidor Web y API de Triaje Clínico (FastAPI + HTML Moderno)
Permite interactuar visualmente con el sistema de triaje clínico inteligente.
"""

import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from .evaluador_integral import EvaluadorTriajeIntegral

app = FastAPI(
    title="Sistema Híbrido de Triaje Clínico",
    description="Plataforma de Decisión Clínica basada en Reglas, ML Supervisado, Clustering y Q-Learning",
    version="3.0"
)

raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
evaluador = EvaluadorTriajeIntegral(raiz)


class PacienteInput(BaseModel):
    fiebre: float
    edad: int
    presion_sistolica: float
    presion_diastolica: float
    bmi: float
    glucosa: float
    tos_seca: int


@app.get("/", response_class=HTMLResponse)
def index():
    html_content = """
<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Triaje Clínico Inteligente | Sprints 1, 2 y 3</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
</head>
<body class="bg-slate-50 text-slate-800 min-h-screen">
  <!-- Encabezado -->
  <header class="bg-gradient-to-r from-blue-700 via-indigo-700 to-teal-600 text-white shadow-lg py-6 px-4 sm:px-8">
    <div class="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
      <div class="flex items-center gap-3">
        <div class="w-12 h-12 bg-white/20 rounded-xl flex items-center justify-center text-2xl">
          <i class="fa-solid fa-heart-pulse"></i>
        </div>
        <div>
          <h1 class="text-2xl sm:text-3xl font-bold tracking-tight">Sistema Híbrido de Triaje Clínico</h1>
          <p class="text-blue-100 text-sm">Proyecto Integrador: Motor de Reglas · ML Supervisado · Clustering · Q-Learning</p>
        </div>
      </div>
      <div class="flex items-center gap-2 text-xs bg-white/10 px-3 py-1.5 rounded-full border border-white/20">
        <span class="inline-block w-2.5 h-2.5 bg-green-400 rounded-full animate-pulse"></span>
        <span>Modelos Cargados: SVM (100% Recall) + Q-Table (FN=-50)</span>
      </div>
    </div>
  </header>

  <main class="max-w-7xl mx-auto px-4 sm:px-8 py-8">
    <!-- Presets Rápidos -->
    <div class="bg-white rounded-xl shadow-sm border border-slate-200 p-4 mb-6">
      <div class="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">Casos Clínicos de Prueba Rápida:</div>
      <div class="flex flex-wrap gap-2">
        <button onclick="cargarPreset('critico')" class="px-3 py-1.5 bg-red-50 text-red-700 hover:bg-red-100 border border-red-200 rounded-lg text-sm font-medium transition">
          <i class="fa-solid fa-triangle-exclamation mr-1"></i> Paciente Crítico (Riesgo Alto)
        </button>
        <button onclick="cargarPreset('estable')" class="px-3 py-1.5 bg-emerald-50 text-emerald-700 hover:bg-emerald-100 border border-emerald-200 rounded-lg text-sm font-medium transition">
          <i class="fa-solid fa-shield-heart mr-1"></i> Paciente Sano / Estable
        </button>
        <button onclick="cargarPreset('hipertension')" class="px-3 py-1.5 bg-amber-50 text-amber-700 hover:bg-amber-100 border border-amber-200 rounded-lg text-sm font-medium transition">
          <i class="fa-solid fa-gauge-high mr-1"></i> Crisis Hipertensiva
        </button>
        <button onclick="cargarPreset('respiratoria')" class="px-3 py-1.5 bg-sky-50 text-sky-700 hover:bg-sky-100 border border-sky-200 rounded-lg text-sm font-medium transition">
          <i class="fa-solid fa-head-side-cough mr-1"></i> Infección Respiratoria Leve
        </button>
      </div>
    </div>

    <div class="grid grid-cols-1 lg:grid-cols-12 gap-8">
      <!-- Formulario de Entrada -->
      <div class="lg:col-span-5 bg-white rounded-2xl shadow-sm border border-slate-200 p-6">
        <h2 class="text-lg font-bold text-slate-800 flex items-center gap-2 mb-4">
          <i class="fa-solid fa-user-doctor text-blue-600"></i> Parámetros Vitales del Paciente
        </h2>
        <form id="triajeForm" onsubmit="evaluarPaciente(event)" class="space-y-4">
          <div class="grid grid-cols-2 gap-4">
            <div>
              <label class="block text-xs font-medium text-slate-600 mb-1">Temperatura (°C)</label>
              <input type="number" step="0.1" id="fiebre" required class="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 text-sm font-medium">
            </div>
            <div>
              <label class="block text-xs font-medium text-slate-600 mb-1">Edad (Años)</label>
              <input type="number" id="edad" required class="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 text-sm font-medium">
            </div>
          </div>

          <div class="grid grid-cols-2 gap-4">
            <div>
              <label class="block text-xs font-medium text-slate-600 mb-1">Presión Sistólica (mmHg)</label>
              <input type="number" step="1" id="presion_sistolica" required class="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 text-sm font-medium">
            </div>
            <div>
              <label class="block text-xs font-medium text-slate-600 mb-1">Presión Diastólica (mmHg)</label>
              <input type="number" step="1" id="presion_diastolica" required class="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 text-sm font-medium">
            </div>
          </div>

          <div class="grid grid-cols-2 gap-4">
            <div>
              <label class="block text-xs font-medium text-slate-600 mb-1">IMC (kg/m²)</label>
              <input type="number" step="0.1" id="bmi" required class="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 text-sm font-medium">
            </div>
            <div>
              <label class="block text-xs font-medium text-slate-600 mb-1">Glucosa (mg/dL)</label>
              <input type="number" step="1" id="glucosa" required class="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 text-sm font-medium">
            </div>
          </div>

          <div>
            <label class="block text-xs font-medium text-slate-600 mb-1">¿Presenta Tos Seca?</label>
            <select id="tos_seca" class="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 text-sm font-medium">
              <option value="1">Sí (1)</option>
              <option value="0">No (0)</option>
            </select>
          </div>

          <button type="submit" class="w-full py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-bold shadow-md hover:shadow-lg transition flex items-center justify-center gap-2">
            <i class="fa-solid fa-bolt"></i> Ejecutar Triaje Clínico Integral
          </button>
        </form>
      </div>

      <!-- Resultados del Sistema -->
      <div class="lg:col-span-7 space-y-6">
        <!-- Tarjeta de Decisión Principal -->
        <div id="cardDecision" class="bg-white rounded-2xl shadow-sm border border-slate-200 p-6 transition-all duration-300">
          <div class="flex items-center justify-between border-b pb-4 mb-4">
            <div>
              <div class="text-xs uppercase font-bold text-slate-400">Decisión de Fusión Híbrida (Sprint 2)</div>
              <h3 id="txtDecision" class="text-2xl font-black text-slate-800">Listo para evaluar</h3>
            </div>
            <div id="badgeDecision" class="px-4 py-2 rounded-xl text-sm font-bold bg-slate-100 text-slate-600">
              En espera
            </div>
          </div>

          <!-- Métricas de Score -->
          <div class="grid grid-cols-3 gap-4 text-center">
            <div class="bg-slate-50 p-3 rounded-xl border border-slate-100">
              <div class="text-xs text-slate-500">Probabilidad ML</div>
              <div id="valProbML" class="text-xl font-bold text-slate-800">--</div>
            </div>
            <div class="bg-slate-50 p-3 rounded-xl border border-slate-100">
              <div class="text-xs text-slate-500">Confianza Reglas</div>
              <div id="valConfReglas" class="text-xl font-bold text-slate-800">--</div>
            </div>
            <div class="bg-blue-50 p-3 rounded-xl border border-blue-100">
              <div class="text-xs text-blue-600 font-semibold">Score Híbrido Final</div>
              <div id="valScoreHibrido" class="text-xl font-black text-blue-700">--</div>
            </div>
          </div>
        </div>

        <!-- Módulos de los 3 Sprints -->
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
          <!-- Sprint 1: Reglas -->
          <div class="bg-white rounded-xl shadow-sm border border-slate-200 p-5">
            <div class="flex items-center justify-between mb-3">
              <h4 class="font-bold text-sm text-slate-700 flex items-center gap-1.5">
                <i class="fa-solid fa-list-check text-indigo-500"></i> Sprint 1: Reglas Expertas
              </h4>
              <span id="badgeNumReglas" class="text-xs px-2 py-0.5 bg-slate-100 rounded-full font-semibold">0 activadas</span>
            </div>
            <div id="listaReglas" class="space-y-2 text-xs text-slate-600 max-h-40 overflow-y-auto pr-1">
              <p class="text-slate-400 italic">No hay reglas evaluadas aún.</p>
            </div>
          </div>

          <!-- Sprint 3B: Q-Learning -->
          <div class="bg-white rounded-xl shadow-sm border border-slate-200 p-5">
            <div class="flex items-center justify-between mb-3">
              <h4 class="font-bold text-sm text-slate-700 flex items-center gap-1.5">
                <i class="fa-solid fa-robot text-teal-500"></i> Sprint 3: Q-Learning
              </h4>
              <span id="badgeEstadoMDP" class="text-xs px-2 py-0.5 bg-slate-100 rounded-full font-semibold">MDP: --</span>
            </div>
            <div class="bg-slate-50 p-3 rounded-lg border border-slate-100 mb-2">
              <div class="text-xs text-slate-500">Acción Óptima Aprendida (π*):</div>
              <div id="txtAccionQL" class="text-base font-black text-teal-700">--</div>
            </div>
            <div id="valoresQ" class="text-[11px] text-slate-500 space-y-1">
              <div>Valores Q: Alta: -, Obs: -, Urg: -</div>
            </div>
          </div>
        </div>

        <!-- Sprint 3A: No Supervisado -->
        <div class="bg-white rounded-xl shadow-sm border border-slate-200 p-5">
          <h4 class="font-bold text-sm text-slate-700 flex items-center gap-1.5 mb-2">
            <i class="fa-solid fa-diagram-project text-purple-500"></i> Sprint 3: Agrupamiento No Supervisado (K-Means & SOM)
          </h4>
          <div class="grid grid-cols-2 gap-4 text-xs">
            <div class="bg-purple-50/50 p-3 rounded-lg border border-purple-100">
              <span class="text-slate-500 font-medium">Clúster K-Means Asignado:</span>
              <div id="txtClusterKM" class="font-bold text-purple-800 text-sm mt-0.5">--</div>
            </div>
            <div class="bg-purple-50/50 p-3 rounded-lg border border-purple-100">
              <span class="text-slate-500 font-medium">Neurona Ganadora SOM (7x7):</span>
              <div id="txtCoordSOM" class="font-bold text-purple-800 text-sm mt-0.5">--</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </main>

  <script>
    const presets = {
      critico: { fiebre: 39.2, edad: 62, presion_sistolica: 150, presion_diastolica: 95, bmi: 31.0, glucosa: 190, tos_seca: 1 },
      estable: { fiebre: 36.5, edad: 28, presion_sistolica: 115, presion_diastolica: 75, bmi: 22.0, glucosa: 85, tos_seca: 0 },
      hipertension: { fiebre: 36.8, edad: 55, presion_sistolica: 165, presion_diastolica: 102, bmi: 27.5, glucosa: 105, tos_seca: 0 },
      respiratoria: { fiebre: 37.8, edad: 24, presion_sistolica: 118, presion_diastolica: 78, bmi: 23.0, glucosa: 90, tos_seca: 1 }
    };

    function cargarPreset(tipo) {
      const p = presets[tipo];
      if (!p) return;
      document.getElementById('fiebre').value = p.fiebre;
      document.getElementById('edad').value = p.edad;
      document.getElementById('presion_sistolica').value = p.presion_sistolica;
      document.getElementById('presion_diastolica').value = p.presion_diastolica;
      document.getElementById('bmi').value = p.bmi;
      document.getElementById('glucosa').value = p.glucosa;
      document.getElementById('tos_seca').value = p.tos_seca;
      evaluarPaciente(new Event('submit'));
    }

    async function evaluarPaciente(e) {
      if (e) e.preventDefault();
      const payload = {
        fiebre: parseFloat(document.getElementById('fiebre').value),
        edad: parseInt(document.getElementById('edad').value),
        presion_sistolica: parseFloat(document.getElementById('presion_sistolica').value),
        presion_diastolica: parseFloat(document.getElementById('presion_diastolica').value),
        bmi: parseFloat(document.getElementById('bmi').value),
        glucosa: parseFloat(document.getElementById('glucosa').value),
        tos_seca: parseInt(document.getElementById('tos_seca').value)
      };

      try {
        const resp = await fetch('/api/evaluar', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await resp.json();
        actualizarUI(data);
      } catch (err) {
        alert('Error conectando con el motor de triaje: ' + err);
      }
    }

    function actualizarUI(d) {
      const s2 = d.sprint_2_supervisado_hibrido;
      const s1 = d.sprint_1_reglas;
      const s3a = d.sprint_3_no_supervisado;
      const s3b = d.sprint_3_refuerzo;

      // Tarjeta principal
      const esUrgente = s2.decision_final_triaje.includes('URGENCIA');
      document.getElementById('txtDecision').innerText = s2.decision_final_triaje;
      const badge = document.getElementById('badgeDecision');
      if (esUrgente) {
        badge.className = 'px-4 py-2 rounded-xl text-sm font-bold bg-red-100 text-red-700 border border-red-200';
        badge.innerText = 'ATENCIÓN INMEDIATA';
      } else {
        badge.className = 'px-4 py-2 rounded-xl text-sm font-bold bg-emerald-100 text-emerald-700 border border-emerald-200';
        badge.innerText = 'NO URGENTE / ESTABLE';
      }

      document.getElementById('valProbML').innerText = (s2.probabilidad_riesgo_ml * 100).toFixed(1) + '%';
      document.getElementById('valConfReglas').innerText = (s1.confianza_maxima * 100).toFixed(1) + '%';
      document.getElementById('valScoreHibrido').innerText = (s2.score_hibrido * 100).toFixed(1) + '%';

      // Sprint 1 Reglas
      document.getElementById('badgeNumReglas').innerText = s1.reglas_activadas + ' activadas';
      const divLista = document.getElementById('listaReglas');
      if (s1.detalle_reglas.length === 0) {
        divLista.innerHTML = '<p class="text-slate-400 italic">Ninguna regla clínica de alerta disparada.</p>';
      } else {
        divLista.innerHTML = s1.detalle_reglas.map(r => `
          <div class="p-2 bg-slate-50 border border-slate-100 rounded-lg">
            <div class="font-bold text-slate-700 flex justify-between">
              <span>[${r.regla_id}] ${r.conclusion}</span>
              <span class="text-blue-600">${(r.confianza * 100).toFixed(0)}%</span>
            </div>
            <div class="text-[11px] text-slate-500">${r.descripcion}</div>
          </div>
        `).join('');
      }

      // Sprint 3A No Supervisado
      document.getElementById('txtClusterKM').innerText = `Clúster ${s3a.cluster_kmeans}: ${s3a.cluster_interpretacion}`;
      document.getElementById('txtCoordSOM').innerText = `(${s3a.coordenadas_som[0]}, ${s3a.coordenadas_som[1]}) en cuadrícula 7x7`;

      // Sprint 3B Q-Learning
      document.getElementById('badgeEstadoMDP').innerText = 'MDP: ' + s3b.estado_mdp;
      document.getElementById('txtAccionQL').innerText = s3b.accion_optima_qlearning.toUpperCase();
      const q = s3b.valores_q_estado;
      document.getElementById('valoresQ').innerHTML = `
        <span class="mr-2">Alta: <b>${q['Alta a casa']}</b></span>
        <span class="mr-2">Observación: <b>${q['Observación']}</b></span>
        <span>Urgencia: <b>${q['Urgencia inmediata']}</b></span>
      `;
    }

    // Cargar por defecto el preset crítico al abrir
    window.onload = () => cargarPreset('critico');
  </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)


@app.post("/api/evaluar")
def api_evaluar(paciente: PacienteInput):
    resultado = evaluador.evaluar_paciente(paciente.dict())
    return JSONResponse(content=resultado)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.app_web:app", host="0.0.0.0", port=8000, reload=False)
