// Laboratorio de evaluación humana ciega, en formato paso a paso para personas sin contexto técnico.
// Protocolo intacto: escala 0–3, restricción, evidencia insuficiente, nota y orden aleatorio por
// evaluador; el backend sigue exigiendo juzgar todo el pool de la consulta.
import { api } from "./api";
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
  has_constraint: boolean;
  destinations: Destination[];
}
interface Answer {
  relevance?: number;
  constraint: string;
  insufficient: boolean;
  note: string;
}

const SCORES = [
  { value: 0, label: "Nada", hint: "No tiene que ver" },
  { value: 1, label: "Poco", hint: "Algo, pero flojo" },
  { value: 2, label: "Sí", hint: "Encaja bien" },
  { value: 3, label: "Perfecto", hint: "Justo lo que pide" },
];
const CONSTRAINT = [
  ["cumple", "Sí la respeta"], ["incumple", "No la respeta"], ["no_se", "No sé"],
];
const TYPE_LABEL: Record<string, string> = { Hotel: "Hotel", Restaurant: "Restaurante", Attractive: "Atractivo" };
const NAME_KEY = "orbita-evaluador";
// Palabras frecuentes que no describen el lugar (“gran”, “servicio”): ruido para quien evalúa.
const GENERIC = new Set(("gran mejor buen buena bueno excelente increible bonito bonita lugar lugares servicio " +
  "atencion recomiendo recomendable vale pena bien muy todo visitar visita casa").split(" "));
const fold = (t: string) => t.normalize("NFD").replace(/[\u0300-\u036f]/g, "").toLowerCase();
const topics = (d: Destination) => {
  const own = new Set(fold(`${d.name} ${d.region}`).split(/\s+/));
  return d.topics.filter((t) => !t.split(" ").some((w) => GENERIC.has(fold(w)) || own.has(fold(w)))).slice(0, 10);
};
const plural = (n: number, one: string, many: string) => `${n} ${n === 1 ? one : many}`;
const fmt = (v: unknown) => (typeof v === "number" ? v.toFixed(2) : v === null || v === undefined ? "—" : String(v));

const store = {
  get: (k: string) => { try { return localStorage.getItem(k) ?? ""; } catch { return ""; } },
  set: (k: string, v: string) => { try { localStorage.setItem(k, v); } catch { /* modo privado */ } },
};

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(api(path));
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `Error ${response.status}`);
  return data as T;
}

export function setupLab(root: HTMLElement) {
  const stage = root.querySelector<HTMLElement>("#lab-stage")!;
  const reportBox = root.querySelector<HTMLElement>("#lab-report")!;
  const counter = root.querySelector<HTMLElement>("#lab-counter")!;
  let evaluator = store.get(NAME_KEY);
  let split = "test";
  let task: Task | null = null;
  let answers: Answer[] = [];
  let index = 0;
  let started = 0;
  let done = 0;

  const error = (message: string) =>
    `<div class="lab-card lab-center"><p class="lab-error" role="alert">${escapeHtml(message)}</p>
      <button type="button" class="lab-primary" data-act="retry">Intentar de nuevo</button></div>`;

  function welcome() {
    task = null;
    stage.innerHTML = `
      <div class="lab-card lab-welcome">
        <div class="lab-steps-mini">
          <div><b>1</b><span>Lees lo que busca un viajero</span></div>
          <div><b>2</b><span>Ves un pueblo y sus reseñas</span></div>
          <div><b>3</b><span>Tocas qué tan bien encaja</span></div>
        </div>
        <p class="lab-lede">Unos <b>2 minutos por consulta</b>. No hay respuestas correctas: queremos tu opinión honesta.
          No te decimos qué sistema propuso cada pueblo, para que tu juicio sea imparcial.</p>
        <form class="lab-start" id="lab-start">
          <label for="lab-name">¿Cómo te llamamos?</label>
          <div class="lab-start-row">
            <input id="lab-name" type="text" maxlength="60" autocomplete="nickname" required
              placeholder="Tu nombre, iniciales o un apodo" value="${escapeHtml(evaluator)}" />
            <button type="submit" class="lab-primary">Empezar <span aria-hidden="true">→</span></button>
          </div>
          <p class="lab-fine">Solo se guarda este alias y tus calificaciones. Si regresas con el mismo alias, continúas donde te quedaste.</p>
          <details class="lab-advanced"><summary>Opciones del equipo</summary>
            <label>Conjunto de consultas
              <select id="lab-split">
                <option value="test" ${split === "test" ? "selected" : ""}>Evaluación final (recomendado)</option>
                <option value="dev" ${split === "dev" ? "selected" : ""}>Desarrollo</option>
                <option value="calibration" ${split === "calibration" ? "selected" : ""}>Calibración</option>
              </select></label>
          </details>
        </form>
      </div>`;
    stage.querySelector<HTMLFormElement>("#lab-start")!.addEventListener("submit", (event) => {
      event.preventDefault();
      evaluator = stage.querySelector<HTMLInputElement>("#lab-name")!.value.trim();
      split = stage.querySelector<HTMLSelectElement>("#lab-split")!.value;
      if (!evaluator) return;
      store.set(NAME_KEY, evaluator);
      next();
    });
  }

  async function next() {
    stage.innerHTML = `<div class="lab-card lab-center"><div class="lab-spinner" aria-hidden="true"></div><p>Preparando una consulta…</p></div>`;
    try {
      const { queue } = await getJson<{ queue: { query_id: string }[] }>(
        `/api/lab/queue?evaluator=${encodeURIComponent(evaluator)}&split=${split}`);
      if (!queue.length) { finished(); return; }
      task = await getJson<Task>(
        `/api/lab/task?query_id=${encodeURIComponent(queue[0].query_id)}&evaluator=${encodeURIComponent(evaluator)}`);
      answers = task.destinations.map(() => ({ constraint: "no_se", insufficient: false, note: "" }));
      index = 0;
      started = performance.now();
      intro();
    } catch (e) {
      stage.innerHTML = error(e instanceof Error ? e.message : "No se pudo conectar con el motor.");
    }
  }

  function intro() {
    if (!task) return;
    stage.innerHTML = `
      <div class="lab-card lab-center lab-brief">
        <p class="lab-kicker">Un viajero escribió:</p>
        <blockquote>“${escapeHtml(task.query)}”</blockquote>
        <p>Te mostraremos <b>${task.destinations.length} pueblos</b>, uno por uno. Para cada uno, piensa:
          <i>¿le recomendaría este lugar a esta persona?</i></p>
        <p class="lab-fine">Califica qué tan bien <b>encaja con lo que pide</b>, no si el pueblo es bonito en general.</p>
        <button type="button" class="lab-primary" data-act="begin">Ver el primer pueblo <span aria-hidden="true">→</span></button>
      </div>`;
  }

  function card() {
    if (!task) return;
    const d = task.destinations[index];
    const a = answers[index];
    const total = task.destinations.length;
    const pct = Math.round((index / total) * 100);
    stage.innerHTML = `
      <div class="lab-progress" aria-label="Progreso"><i style="width:${pct}%"></i></div>
      <div class="lab-meta"><span>Pueblo ${index + 1} de ${total}</span><span class="lab-q">“${escapeHtml(task.query)}”</span></div>
      <article class="lab-card lab-dest" tabindex="-1" aria-live="polite">
        <header><p class="lab-region">${escapeHtml(d.region)}</p><h3>${escapeHtml(d.name)}</h3></header>
        <p class="lab-label">De qué hablan sus reseñas</p>
        <ul class="lab-chips">${topics(d).map((t) => `<li>${escapeHtml(t)}</li>`).join("")}</ul>
        <details class="lab-reviews">
          <summary>Leer ${d.sample_reviews.length} reseñas de visitantes</summary>
          <ul>${d.sample_reviews.map((r) => `<li><small>${escapeHtml(TYPE_LABEL[r.type] ?? r.type)}</small>${escapeHtml(r.text)}…</li>`).join("")}</ul>
        </details>
        ${task.has_constraint ? `
          <fieldset class="lab-constraint"><legend>La consulta pide evitar algo. ¿Este pueblo lo respeta?</legend>
            ${CONSTRAINT.map(([v, l]) => `<button type="button" data-con="${v}" aria-pressed="${a.constraint === v}">${l}</button>`).join("")}
          </fieldset>` : ""}
        <p class="lab-ask">¿Qué tan bien encaja con lo que pide?</p>
        <div class="lab-scores" role="group" aria-label="Calificación">
          ${SCORES.map((s) => `
            <button type="button" class="score-${s.value}" data-score="${s.value}" aria-pressed="${a.relevance === s.value}">
              <kbd>${s.value}</kbd><b>${s.label}</b><small>${s.hint}</small></button>`).join("")}
        </div>
        <div class="lab-foot">
          <button type="button" class="lab-link" data-act="back" ${index === 0 ? "disabled" : ""}>← Anterior</button>
          <label class="lab-check"><input type="checkbox" data-ins ${a.insufficient ? "checked" : ""} /> No hay suficiente información para juzgar</label>
          <details class="lab-note"><summary>Comentario</summary>
            <input type="text" maxlength="500" data-note value="${escapeHtml(a.note)}" placeholder="Opcional" /></details>
        </div>
      </article>
      <p class="lab-keys">Atajo: teclas <kbd>0</kbd> <kbd>1</kbd> <kbd>2</kbd> <kbd>3</kbd> para calificar · <kbd>←</kbd> para regresar</p>`;
    stage.querySelector<HTMLElement>(".lab-dest")?.focus({ preventScroll: true });
  }

  function rate(value: number) {
    if (!task) return;
    answers[index].relevance = value;
    stage.querySelector(`[data-score="${value}"]`)?.classList.add("chosen");
    // Pausa breve para que se vea la elección antes de avanzar.
    setTimeout(() => {
      if (index < task!.destinations.length - 1) { index += 1; card(); } else { review(); }
    }, 160);
  }

  function review() {
    if (!task) return;
    const missing = answers.findIndex((a) => a.relevance === undefined);
    stage.innerHTML = `
      <div class="lab-progress"><i style="width:100%"></i></div>
      <div class="lab-card">
        <h3 class="lab-title">¡Listo! Revisa antes de enviar</h3>
        <p class="lab-fine">Toca cualquier pueblo para cambiar su calificación.</p>
        <ol class="lab-summary">${task.destinations.map((d, i) => {
          const r = answers[i].relevance;
          return `<li><button type="button" data-goto="${i}">
            <span>${escapeHtml(d.name)}</span>
            <b class="pill score-${r ?? "x"}">${r === undefined ? "Falta" : `${r} · ${SCORES[r].label}`}</b></button></li>`;
        }).join("")}</ol>
        ${missing >= 0 ? `<p class="lab-error">Falta calificar ${answers.filter((a) => a.relevance === undefined).length} pueblo(s).</p>` : ""}
        <button type="button" class="lab-primary" data-act="send" ${missing >= 0 ? "disabled" : ""}>Enviar mis calificaciones ✓</button>
      </div>`;
  }

  async function send() {
    if (!task) return;
    const button = stage.querySelector<HTMLButtonElement>('[data-act="send"]');
    if (button) { button.disabled = true; button.textContent = "Enviando…"; }
    try {
      const saved = await postJson<any>("/api/lab/judgments", {
        evaluator, query_id: task.query_id,
        seconds: Math.round((performance.now() - started) / 100) / 10,
        judgments: task.destinations.map((d, i) => ({
          town: d.town,
          relevance: answers[i].relevance,
          constraint: task!.has_constraint ? answers[i].constraint : "no_aplica",
          insufficient_evidence: answers[i].insufficient,
          note: answers[i].note,
        })),
      });
      done += 1;
      renderReport(saved.report);
      stage.innerHTML = `
        <div class="lab-card lab-center lab-thanks">
          <div class="lab-burst" aria-hidden="true">✓</div>
          <h3>¡Gracias, ${escapeHtml(evaluator)}!</h3>
          <p>Guardamos ${saved.saved} calificaciones. Llevas <b>${done}</b> consulta${done === 1 ? "" : "s"} en esta visita.</p>
          <p class="lab-fine">Cada consulta extra nos ayuda a medir el acuerdo entre personas (alfa de Krippendorff).</p>
          <div class="lab-row">
            <button type="button" class="lab-primary" data-act="next">Evaluar otra consulta →</button>
            <button type="button" class="lab-link" data-act="stop">Terminar por hoy</button>
          </div>
        </div>`;
      task = null;
    } catch (e) {
      if (button) { button.disabled = false; button.textContent = "Enviar mis calificaciones ✓"; }
      stage.querySelector(".lab-card")?.insertAdjacentHTML("beforeend",
        `<p class="lab-error" role="alert">${escapeHtml(e instanceof Error ? e.message : "No se pudo guardar.")}</p>`);
    }
  }

  function finished() {
    stage.innerHTML = `<div class="lab-card lab-center lab-thanks"><div class="lab-burst" aria-hidden="true">★</div>
      <h3>¡Completaste todas las consultas!</h3><p>No quedan consultas pendientes para ${escapeHtml(evaluator)} en este conjunto. Muchas gracias.</p>
      <button type="button" class="lab-link" data-act="stop">Volver al inicio</button></div>`;
  }

  function renderReport(report: any) {
    const a = report.agreement ?? {};
    const humans = document.querySelector("#human-count");
    if (humans) humans.textContent = report.evaluators ? plural(report.evaluators, "persona", "personas") : "Abierto";
    counter.textContent = report.judgments
      ? `${plural(report.judgments, "calificación", "calificaciones")} · ${plural(report.evaluators, "persona", "personas")} · ${plural(report.queries_judged, "consulta", "consultas")}`
      : "Sé de las primeras personas en evaluar Órbita.";
    const splits = Object.entries(report.by_split_and_system ?? {});
    reportBox.innerHTML = `
      <p class="eval-note">Acuerdo entre personas (ítems con ≥2 evaluadores: ${a.items_with_2plus_raters ?? 0}):
        α ordinal ${fmt(a.krippendorff_alpha_ordinal)} · α restricción ${fmt(a.constraint_alpha_nominal)}
        ${a.sufficient ? "" : " · aún insuficiente (se recomiendan ≥30 ítems compartidos)"} ·
        mediana ${fmt(report.median_seconds_per_query)} s por consulta.</p>
      ${splits.map(([name, systems]: [string, any]) => `
        <table><caption>Conjunto ${escapeHtml(name)}</caption>
          <thead><tr><th scope="col">Sistema</th><th scope="col">nDCG@5</th><th scope="col">P@5</th>
            <th scope="col">Relevancia media bottom 5</th><th scope="col">Errores de restricción top 5</th></tr></thead>
          <tbody>${Object.entries(systems).map(([sys, m]: [string, any]) => `<tr><th scope="row">${escapeHtml(sys)}</th>
            <td>${fmt(m.ndcg_at_5)}</td><td>${fmt(m.precision_at_5)}</td><td>${fmt(m.bottom5_mean_relevance)}</td>
            <td>${fmt(m.top5_constraint_errors)}</td></tr>`).join("")}</tbody></table>`).join("")}
      <p><a href="${api("/api/lab/export?format=csv")}">Exportar CSV</a> · <a href="${api("/api/lab/export?format=json")}">Exportar JSON</a></p>`;
  }

  stage.addEventListener("click", (event) => {
    const el = (event.target as HTMLElement).closest<HTMLElement>("button");
    if (!el || !task && !el.dataset.act) return;
    if (el.dataset.score !== undefined) rate(Number(el.dataset.score));
    else if (el.dataset.con && task) {
      answers[index].constraint = el.dataset.con;
      el.parentElement!.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b === el)));
    } else if (el.dataset.goto !== undefined) { index = Number(el.dataset.goto); card(); }
    else switch (el.dataset.act) {
      case "begin": card(); break;
      case "back": if (index > 0) { index -= 1; card(); } break;
      case "send": send(); break;
      case "next": case "retry": next(); break;
      case "stop": welcome(); break;
    }
  });
  stage.addEventListener("change", (event) => {
    const el = event.target as HTMLInputElement;
    if (!task) return;
    if (el.matches("[data-ins]")) answers[index].insufficient = el.checked;
  });
  stage.addEventListener("input", (event) => {
    const el = event.target as HTMLInputElement;
    if (task && el.matches("[data-note]")) answers[index].note = el.value;
  });
  document.addEventListener("keydown", (event) => {
    if (!task || !stage.querySelector(".lab-dest")) return;
    const box = stage.getBoundingClientRect();
    if (box.bottom < 0 || box.top > window.innerHeight) return; // solo con el laboratorio en pantalla
    const target = event.target as HTMLElement;
    if (target.matches("input[type=text], textarea, select") || event.metaKey || event.ctrlKey) return;
    if (/^[0-3]$/.test(event.key)) { event.preventDefault(); rate(Number(event.key)); }
    else if (event.key === "ArrowLeft" && index > 0) { event.preventDefault(); index -= 1; card(); }
  });

  welcome();
  getJson<any>("/api/lab/report").then(renderReport).catch(() => {
    counter.textContent = "El laboratorio necesita el motor en línea.";
  });
}
