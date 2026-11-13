// Laboratorio de evaluación humana ciega: escala 0–3, restricción, evidencia insuficiente y notas.
import { escapeHtml, postJson } from "./classic";

interface Destination {
  town: string;
  name: string;
  region: string;
  topics: string[];
  sample_reviews: { type: string; text: string }[];
}
interface Task {
  query_id: string;
  query: string;
  split: string;
  has_constraint: boolean;
  destinations: Destination[];
  scale: Record<string, string>;
}

const fmt = (v: unknown) => (typeof v === "number" ? v.toFixed(3) : v === null || v === undefined ? "—" : String(v));

export function setupLab(root: HTMLElement) {
  const evaluator = root.querySelector<HTMLInputElement>("#lab-evaluator")!;
  const split = root.querySelector<HTMLSelectElement>("#lab-split")!;
  const queueBox = root.querySelector<HTMLElement>("#lab-queue")!;
  const stage = root.querySelector<HTMLElement>("#lab-stage")!;
  const status = root.querySelector<HTMLElement>("#lab-status")!;
  const reportBox = root.querySelector<HTMLElement>("#lab-report")!;
  let task: Task | null = null;
  let started = 0;

  const who = () => evaluator.value.trim();

  async function loadQueue() {
    if (!who()) { status.textContent = "Escribe un identificador de evaluador (p. ej. tus iniciales)."; evaluator.focus(); return; }
    status.textContent = "Cargando cola…";
    try {
      const response = await fetch(`/api/lab/queue?evaluator=${encodeURIComponent(who())}&split=${split.value}`);
      const data = await response.json();
      if (!response.ok) throw new Error(data.error);
      if (!data.queue.length) { queueBox.innerHTML = "<p>No quedan consultas pendientes en esta partición.</p>"; status.textContent = ""; return; }
      queueBox.innerHTML = `<ol class="lab-queue">${data.queue.map((q: any) =>
        `<li><button type="button" class="link-button" data-qid="${escapeHtml(q.query_id)}">${escapeHtml(q.query)}</button>
         <small>${escapeHtml(q.reason)}</small></li>`).join("")}</ol>`;
      queueBox.querySelectorAll<HTMLButtonElement>("[data-qid]").forEach((b) =>
        b.addEventListener("click", () => openTask(b.dataset.qid!)));
      status.textContent = `${data.queue.length} consultas en cola.`;
    } catch (error) {
      status.textContent = error instanceof Error ? error.message : "Error desconocido.";
    }
  }

  async function openTask(queryId: string) {
    status.textContent = "Preparando destinos…";
    const response = await fetch(`/api/lab/task?query_id=${encodeURIComponent(queryId)}&evaluator=${encodeURIComponent(who())}`);
    const data = await response.json();
    if (!response.ok) { status.textContent = data.error; return; }
    task = data as Task;
    started = performance.now();
    render();
    status.textContent = `Consulta “${task.query}” · ${task.destinations.length} destinos en orden aleatorio.`;
    stage.querySelector<HTMLElement>("fieldset input")?.focus();
  }

  function render() {
    if (!task) return;
    const scale = Object.entries(task.scale);
    stage.innerHTML = `
      <p class="lab-query">Consulta: <b>${escapeHtml(task.query)}</b></p>
      <p class="lab-help">Califica qué tan compatible es cada destino <i>con esta consulta</i> (no su calidad general).
        Teclas <kbd>0</kbd>–<kbd>3</kbd> califican el destino enfocado y pasan al siguiente.</p>
      <form id="lab-form">
        <ol class="lab-items">${task.destinations.map((d, i) => `
          <li class="lab-item" data-town="${escapeHtml(d.town)}">
            <h4>${escapeHtml(d.name)} <small>${escapeHtml(d.region)}</small></h4>
            <p class="lab-topics">Temas frecuentes en sus reseñas: ${escapeHtml(d.topics.join(", "))}</p>
            <details><summary>Reseñas de muestra (no dependen de la consulta)</summary>
              <ul>${d.sample_reviews.map((r) => `<li><small>${escapeHtml(r.type)}</small> ${escapeHtml(r.text)}</li>`).join("")}</ul>
            </details>
            <fieldset class="lab-scale">
              <legend>Compatibilidad de ${escapeHtml(d.name)}</legend>
              ${scale.map(([value, label]) => `
                <label><input type="radio" name="rel-${i}" value="${value}" required /> ${value} · ${escapeHtml(label)}</label>`).join("")}
            </fieldset>
            <div class="lab-extra">
              ${task!.has_constraint ? `<label>Restricción
                <select name="con-${i}">
                  <option value="no_se">No sé</option><option value="cumple">Cumple</option>
                  <option value="incumple">Incumple</option><option value="no_aplica">No aplica</option>
                </select></label>` : ""}
              <label><input type="checkbox" name="ins-${i}" /> Evidencia insuficiente para juzgar</label>
              <label class="lab-note">Nota <input type="text" name="note-${i}" maxlength="500" /></label>
            </div>
          </li>`).join("")}</ol>
        <button type="submit" class="submit-button"><span>Guardar juicios</span><b>✓</b></button>
      </form>`;
    const form = stage.querySelector<HTMLFormElement>("#lab-form")!;
    form.addEventListener("submit", (event) => { event.preventDefault(); save(form); });
    form.addEventListener("keydown", (event) => {
      const target = event.target as HTMLElement;
      if (target.matches("input[type=text], select") || !/^[0-3]$/.test(event.key)) return;
      const item = target.closest<HTMLElement>(".lab-item");
      if (!item) return;
      event.preventDefault();
      item.querySelector<HTMLInputElement>(`input[type=radio][value="${event.key}"]`)!.checked = true;
      const next = item.nextElementSibling?.querySelector<HTMLInputElement>("input[type=radio]");
      (next ?? form.querySelector<HTMLButtonElement>("button[type=submit]"))!.focus();
    });
  }

  async function save(form: HTMLFormElement) {
    if (!task) return;
    const data = new FormData(form);
    const judgments = task.destinations.map((d, i) => ({
      town: d.town,
      relevance: Number(data.get(`rel-${i}`)),
      constraint: task!.has_constraint ? String(data.get(`con-${i}`)) : "no_aplica",
      insufficient_evidence: data.get(`ins-${i}`) === "on",
      note: String(data.get(`note-${i}`) ?? ""),
    }));
    try {
      const saved = await postJson<any>("/api/lab/judgments", {
        evaluator: who(), query_id: task.query_id, judgments,
        seconds: Math.round((performance.now() - started) / 100) / 10,
      });
      stage.innerHTML = `<p class="blind-result" role="status">Guardados ${saved.saved} juicios.
        Las posiciones del sistema se revelan solo en el reporte agregado.</p>`;
      task = null;
      renderReport(saved.report);
      loadQueue();
    } catch (error) {
      status.textContent = error instanceof Error ? error.message : "Error desconocido.";
    }
  }

  function renderReport(report: any) {
    const a = report.agreement;
    const splits = Object.entries(report.by_split_and_system ?? {});
    reportBox.innerHTML = `
      <p class="eval-note">${report.judgments} juicios · ${report.evaluators} evaluadores · ${report.queries_judged} consultas ·
        mediana ${fmt(report.median_seconds_per_query)} s por consulta.</p>
      <p class="eval-note">Acuerdo (ítems con ≥2 evaluadores: ${a.items_with_2plus_raters}):
        α ordinal ${fmt(a.krippendorff_alpha_ordinal)} · α restricción ${fmt(a.constraint_alpha_nominal)}
        ${a.sufficient ? "" : " · aún insuficiente (se recomiendan ≥30 ítems compartidos)"}.</p>
      ${splits.map(([name, systems]: [string, any]) => `
        <table><caption>Partición ${escapeHtml(name)}</caption>
          <thead><tr><th scope="col">Sistema</th><th scope="col">nDCG@5</th><th scope="col">P@5</th>
            <th scope="col">Relevancia media bottom 5</th><th scope="col">Errores de restricción top 5</th></tr></thead>
          <tbody>${Object.entries(systems).map(([sys, m]: [string, any]) => `<tr><th scope="row">${escapeHtml(sys)}</th>
            <td>${fmt(m.ndcg_at_5)}</td><td>${fmt(m.precision_at_5)}</td><td>${fmt(m.bottom5_mean_relevance)}</td>
            <td>${fmt(m.top5_constraint_errors)}</td></tr>`).join("")}</tbody></table>`).join("")}
      <p><a href="/api/lab/export?format=csv">Exportar CSV</a> · <a href="/api/lab/export?format=json">Exportar JSON</a></p>`;
  }

  root.querySelector("#lab-load")!.addEventListener("click", loadQueue);
  fetch("/api/lab/report").then((r) => r.json()).then(renderReport).catch(() => {});
}
