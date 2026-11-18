"""
Servidor Web y API de Triaje Clínico Multimodal (FastAPI + HTML Moderno)
Permite interactuar visualmente con el sistema de triaje clínico inteligente (Sprints 1, 2, 3 y 4).
"""

import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional
from .evaluador_integral import EvaluadorTriajeIntegral

app = FastAPI(
    title="Sistema Multimodal de Triaje Clínico",
    description="Plataforma de Decisión Clínica Multimodal: Reglas, ML Supervisado, Clustering, Q-Learning y CNN",
    version="4.0"
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
    radiografia: Optional[str] = "paciente_critico_neumonia.png"


@app.get("/", response_class=HTMLResponse)
def index():
    html_content = """
<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Triaje Multimodal Inteligente | Sprints 1, 2, 3 y 4</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
</head>
<body class="bg-slate-50 text-slate-800 min-h-screen">
  <!-- Encabezado -->
  <header class="bg-gradient-to-r from-blue-700 via-indigo-700 to-teal-600 text-white shadow-lg py-6 px-4 sm:px-8">
    <div class="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
      <div class="flex items-center gap-3">
        <div class="w-12 h-12 bg-white/20 rounded-xl flex items-center justify-center text-2xl">
          <i class="fa-solid fa-hospital-user"></i>
        </div>
        <div>
          <h1 class="text-2xl sm:text-3xl font-bold tracking-tight">Sistema Multimodal de Triaje Clínico</h1>
          <p class="text-blue-100 text-sm">Integrador: Reglas · ML Supervisado · K-Means & SOM · Q-Learning · CNN Tórax</p>
        </div>
      </div>
      <div class="flex items-center gap-2 text-xs bg-white/10 px-3 py-1.5 rounded-full border border-white/20">
        <span class="inline-block w-2.5 h-2.5 bg-green-400 rounded-full animate-pulse"></span>
        <span>Entrega Final Sprint 4 & Cierre (2 de Diciembre)</span>
      </div>
    </div>
  </header>

  <main class="max-w-7xl mx-auto px-4 sm:px-8 py-8">
    <!-- Presets Rápidos para Defensa -->
    <div class="bg-white rounded-xl shadow-sm border border-slate-200 p-4 mb-6">
      <div class="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">Casos Clínicos para Defensa Oral en Vivo:</div>
      <div class="flex flex-wrap gap-2">
        <button onclick="cargarPreset('critico')" class="px-3 py-1.5 bg-red-50 text-red-700 hover:bg-red-100 border border-red-200 rounded-lg text-sm font-medium transition">
          <i class="fa-solid fa-triangle-exclamation mr-1"></i> Caso 3: Paciente Crítico Multimodal (Alerta Roja)
        </button>
        <button onclick="cargarPreset('limite')" class="px-3 py-1.5 bg-amber-50 text-amber-700 hover:bg-amber-100 border border-amber-200 rounded-lg text-sm font-medium transition">
          <i class="fa-solid fa-scale-balanced mr-1"></i> Caso 2: Caso Límite / Frontera (Alerta Amarilla)
        </button>
        <button onclick="cargarPreset('sano')" class="px-3 py-1.5 bg-emerald-50 text-emerald-700 hover:bg-emerald-100 border border-emerald-200 rounded-lg text-sm font-medium transition">
          <i class="fa-solid fa-shield-heart mr-1"></i> Caso 1: Paciente Sano / Estable (Alerta Verde)
        </button>
      </div>
    </div>

    <div class="grid grid-cols-1 lg:grid-cols-12 gap-8">
      <!-- Formulario de Entrada Multimodal -->
      <div class="lg:col-span-5 bg-white rounded-2xl shadow-sm border border-slate-200 p-6">
        <h2 class="text-lg font-bold text-slate-800 flex items-center gap-2 mb-4">
          <i class="fa-solid fa-user-doctor text-blue-600"></i> Parámetros Vitales y Radiografía
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

          <div class="grid grid-cols-2 gap-4">
            <div>
              <label class="block text-xs font-medium text-slate-600 mb-1">¿Tos Seca?</label>
              <select id="tos_seca" class="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 text-sm font-medium">
                <option value="1">Sí (1)</option>
                <option value="0">No (0)</option>
              </select>
            </div>
            <div>
              <label class="block text-xs font-medium text-slate-600 mb-1">Radiografía de Tórax</label>
              <select id="radiografia" class="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 text-sm font-medium">
                <option value="paciente_critico_neumonia.png">Neumonía Severa (Opacidad Densa)</option>
                <option value="paciente_limite.png">Infiltrado Dudoso / Limítrofe</option>
                <option value="paciente_sano.png">Pulmón Sano / Despejado</option>
              </select>
            </div>
          </div>

          <button type="submit" class="w-full py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-bold shadow-md hover:shadow-lg transition flex items-center justify-center gap-2">
            <i class="fa-solid fa-network-wired"></i> Evaluar Triaje Multimodal Completo
          </button>
        </form>
      </div>

      <!-- Resultados Multimodales -->
      <div class="lg:col-span-7 space-y-6">
        <!-- Dictamen Global -->
        <div id="cardDecision" class="bg-white rounded-2xl shadow-sm border border-slate-200 p-6 transition-all duration-300">
          <div class="flex items-center justify-between border-b pb-4 mb-4">
            <div>
              <div class="text-xs uppercase font-bold text-slate-400">Dictamen Final Multimodal (4 Mundos)</div>
              <h3 id="txtDecision" class="text-2xl font-black text-slate-800">Listo para evaluar</h3>
            </div>
            <div id="badgeDecision" class="px-4 py-2 rounded-xl text-sm font-bold bg-slate-100 text-slate-600">
              En espera
            </div>
          </div>

          <!-- Métricas de Score Multimodal -->
          <div class="grid grid-cols-4 gap-3 text-center">
            <div class="bg-slate-50 p-2.5 rounded-xl border border-slate-100">
              <div class="text-[11px] text-slate-500">ML Tabular</div>
              <div id="valProbML" class="text-lg font-bold text-slate-800">--</div>
            </div>
            <div class="bg-slate-50 p-2.5 rounded-xl border border-slate-100">
              <div class="text-[11px] text-slate-500">Reglas JSON</div>
              <div id="valConfReglas" class="text-lg font-bold text-slate-800">--</div>
            </div>
            <div class="bg-slate-50 p-2.5 rounded-xl border border-slate-100">
              <div class="text-[11px] text-slate-500">CNN Tórax</div>
              <div id="valProbCNN" class="text-lg font-bold text-slate-800">--</div>
            </div>
            <div class="bg-blue-50 p-2.5 rounded-xl border border-blue-100">
              <div class="text-[11px] text-blue-600 font-semibold">Score Global</div>
              <div id="valScoreMulti" class="text-lg font-black text-blue-700">--</div>
            </div>
          </div>

          <div id="txtJustificacion" class="mt-4 text-xs bg-slate-50 p-3 rounded-lg border border-slate-100 text-slate-600 italic">
            Justificación clínica: En espera de evaluación.
          </div>
        </div>

        <!-- Sprint 4: CNN Visión Artificial -->
        <div class="bg-white rounded-xl shadow-sm border border-slate-200 p-5">
          <div class="flex items-center justify-between mb-2">
            <h4 class="font-bold text-sm text-slate-700 flex items-center gap-1.5">
              <i class="fa-solid fa-x-ray text-rose-500"></i> Sprint 4: Visión Artificial con CNN (Tórax)
            </h4>
            <span id="badgeCNN" class="text-xs px-2 py-0.5 rounded-full font-semibold bg-slate-100">--</span>
          </div>
          <div id="dictamenCNN" class="text-xs font-semibold text-slate-700">Sin análisis</div>
          <div id="recomCNN" class="text-xs text-slate-500 mt-1">--</div>
        </div>

        <!-- Módulos Sprints 1, 3A y 3B -->
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
          <!-- Sprint 1: Reglas -->
          <div class="bg-white rounded-xl shadow-sm border border-slate-200 p-4">
            <h5 class="font-bold text-xs text-slate-700 mb-2 flex items-center gap-1.5">
              <i class="fa-solid fa-clipboard-check text-indigo-500"></i> Sprint 1: Reglas Expertas
            </h5>
            <div id="listaReglas" class="space-y-1.5 text-xs max-h-32 overflow-y-auto">
              <p class="text-slate-400 italic">Sin reglas evaluadas</p>
            </div>
          </div>

          <!-- Sprint 3B: Q-Learning -->
          <div class="bg-white rounded-xl shadow-sm border border-slate-200 p-4">
            <h5 class="font-bold text-xs text-slate-700 mb-2 flex items-center gap-1.5">
              <i class="fa-solid fa-robot text-teal-500"></i> Sprint 3: Q-Learning & K-Means
            </h5>
            <div class="text-xs text-slate-600 space-y-1">
              <div>Acción Óptima MDP: <b id="txtAccionQL" class="text-teal-700">--</b></div>
              <div>Clúster K-Means: <b id="txtClusterKM">--</b></div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </main>

  <script>
    const presets = {
      critico: { fiebre: 39.4, edad: 68, presion_sistolica: 162, presion_diastolica: 98, bmi: 32.5, glucosa: 210, tos_seca: 1, radiografia: 'paciente_critico_neumonia.png' },
      limite: { fiebre: 37.6, edad: 48, presion_sistolica: 134, presion_diastolica: 86, bmi: 28.2, glucosa: 155, tos_seca: 1, radiografia: 'paciente_limite.png' },
      sano: { fiebre: 36.6, edad: 26, presion_sistolica: 118, presion_diastolica: 76, bmi: 22.4, glucosa: 88, tos_seca: 0, radiografia: 'paciente_sano.png' }
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
      document.getElementById('radiografia').value = p.radiografia;
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
        tos_seca: parseInt(document.getElementById('tos_seca').value),
        radiografia: document.getElementById('radiografia').value
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
        alert('Error conectando con el motor multimodal: ' + err);
      }
    }

    function actualizarUI(d) {
      const g = d.dictamen_global_multimodal;
      const s1 = d.sprint_1_reglas;
      const s2 = d.sprint_2_supervisado_hibrido;
      const s3a = d.sprint_3_no_supervisado;
      const s3b = d.sprint_3_refuerzo;
      const s4 = d.sprint_4_vision_cnn;

      // Tarjeta principal
      document.getElementById('txtDecision').innerText = g.decision_final_triaje;
      const badge = document.getElementById('badgeDecision');
      if (g.decision_final_triaje.includes('ROJA') || g.decision_final_triaje.includes('URGENCIA')) {
        badge.className = 'px-4 py-2 rounded-xl text-sm font-bold bg-red-100 text-red-700 border border-red-200';
        badge.innerText = 'NIVEL 1: PRIORIDAD INMEDIATA';
      } else if (g.decision_final_triaje.includes('AMARILLA') || g.decision_final_triaje.includes('OBSERVACIÓN')) {
        badge.className = 'px-4 py-2 rounded-xl text-sm font-bold bg-amber-100 text-amber-700 border border-amber-200';
        badge.innerText = 'NIVEL 2: OBSERVACIÓN';
      } else {
        badge.className = 'px-4 py-2 rounded-xl text-sm font-bold bg-emerald-100 text-emerald-700 border border-emerald-200';
        badge.innerText = 'NIVEL 3: ESTABLE';
      }

      document.getElementById('valProbML').innerText = (s2.probabilidad_riesgo_ml * 100).toFixed(1) + '%';
      document.getElementById('valConfReglas').innerText = (s1.confianza_maxima * 100).toFixed(0) + '%';
      document.getElementById('valProbCNN').innerText = s4 ? (s4.probabilidad_neumonia * 100).toFixed(1) + '%' : '--';
      document.getElementById('valScoreMulti').innerText = (g.score_multimodal * 100).toFixed(1) + '%';
      document.getElementById('txtJustificacion').innerText = g.justificacion_clinica;

      // Sprint 4 CNN
      if (s4) {
        document.getElementById('badgeCNN').innerText = 'Severidad: ' + s4.severidad_imagen;
        document.getElementById('badgeCNN').className = s4.severidad_imagen === 'ALTA' ? 'text-xs px-2 py-0.5 rounded-full font-semibold bg-red-100 text-red-700' : (s4.severidad_imagen === 'MODERADA' ? 'text-xs px-2 py-0.5 rounded-full font-semibold bg-amber-100 text-amber-700' : 'text-xs px-2 py-0.5 rounded-full font-semibold bg-emerald-100 text-emerald-700');
        document.getElementById('dictamenCNN').innerText = s4.dictamen_radiologico;
        document.getElementById('recomCNN').innerText = s4.recomendacion_clinica;
      }

      // Sprint 1 Reglas
      const divLista = document.getElementById('listaReglas');
      if (s1.detalle_reglas.length === 0) {
        divLista.innerHTML = '<p class="text-slate-400 italic">Sin reglas activadas</p>';
      } else {
        divLista.innerHTML = s1.detalle_reglas.map(r => `
          <div class="p-1.5 bg-slate-50 border border-slate-100 rounded text-[11px]">
            <b>[${r.regla_id}]</b> ${r.conclusion} <span class="text-blue-600 font-bold">(${(r.confianza * 100).toFixed(0)}%)</span>
          </div>
        `).join('');
      }

      // Sprint 3
      document.getElementById('txtAccionQL').innerText = s3b.accion_optima_qlearning.toUpperCase();
      document.getElementById('txtClusterKM').innerText = `Clúster ${s3a.cluster_kmeans} (${s3a.cluster_interpretacion.split('/')[0]})`;
    }

    window.onload = () => cargarPreset('critico');
  </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)


@app.post("/api/evaluar")
def api_evaluar(paciente: PacienteInput):
    ruta_rx = os.path.join(raiz, "data", "radiografias", paciente.radiografia) if paciente.radiografia else None
    resultado = evaluador.evaluar_paciente(paciente.dict(), ruta_imagen_radiografia=ruta_rx)
    return JSONResponse(content=resultado)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.app_web:app", host="0.0.0.0", port=8000, reload=False)
