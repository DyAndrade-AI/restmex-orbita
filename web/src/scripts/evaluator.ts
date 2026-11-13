// Panel de evaluación: rúbrica automática con un clic y etiquetado ciego por teclado.
import { escapeHtml } from "./classic";

const fmt = (value: unknown) =>
  typeof value === "number" ? (Number.isInteger(value) ? String(value) : value.toFixed(3)) : String(value ?? "—");

export function renderEvaluation(report: any): string {
  const rubric = report.rubric.map((row: any) => `
    <li class="rubric-item ${row.pass ? "pass" : "fail"}">
      <span>${row.pass ? "Cumple" : "No cumple"}</span>
      <b>${escapeHtml(row.check)}</b>
      <strong>${fmt(row.value)}</strong><small>umbral ${fmt(row.threshold)}</small>
    </li>`).join("");
  const keys = ["precision_at_5", "capped_recall_at_5", "hit_at_1", "mrr", "ndcg_at_5", "bottom_error_rate"];
  const g = report.gold.aggregate;
  const b = report.baselines;
  const metricRows = keys.map((key) => `<tr><th scope="row">${key}</th><td>${fmt(g[key])}</td>
    <td>${fmt(b.popularity[key])}</td><td>${fmt(b.random[key])}</td></tr>`).join("");
  const failures = report.gold.queries.filter((q: any) => q.hit_at_5 === 0 || q.relevant_in_bottom > 0);
  const holdout = report.holdout ? `<p class="eval-note">Holdout sin fuga (${report.holdout.evaluated_queries} reseñas):
    Hit@5 <b>${fmt(report.holdout.metrics.hit_at_5)}</b> vs popularidad ${fmt(report.holdout.popularity_baseline.hit_at_5)}
    y azar ${fmt(report.holdout.random_expectation.hit_at_5)}.</p>` : "";
  const human = report.human?.sessions
    ? `<p class="eval-note">Evaluación humana: ${report.human.sessions} sesiones · precision@5 ${fmt(report.human.human_precision_at_5)} ·
       bottom no relacionado ${fmt(report.human.human_bottom_unrelated_rate)} · mediana ${fmt(report.human.median_seconds_per_query)} s/consulta.</p>`
    : `<p class="eval-note">Aún no hay etiquetas humanas: usa el modo ciego de abajo.</p>`;
  return `
    <p class="eval-meta">Modo <b>${escapeHtml(report.mode)}</b> · ${report.gold.aggregate.queries} consultas del seed heurístico
      (pendiente de adjudicación humana) · latencia p95 ${fmt(report.gold.latency_s.p95)} s</p>
    <ul class="rubric">${rubric}</ul>
    <div class="eval-grid">
      <table>
        <caption>Sistema frente a referencias</caption>
        <thead><tr><th scope="col">Métrica</th><th scope="col">Sistema</th><th scope="col">Popularidad</th><th scope="col">Azar</th></tr></thead>
        <tbody>${metricRows}</tbody>
      </table>
      <div>
        <p class="eval-note">Adversariales: ${fmt(report.adversarial.pass_rate * 100)}% correctos ·
          determinismo ${fmt(report.determinism.identical_rate * 100)}% ·
          sesgo por volumen (Spearman) ${fmt(report.volume_bias.spearman_exposure_vs_reviews)}.</p>
        ${holdout}${human}
        <h4>Consultas a revisar (${failures.length})</h4>
        <ul class="failures">${failures.map((q: any) => `<li><b>${escapeHtml(q.query)}</b>
          <span>top: ${escapeHtml(q.top.join(", "))}</span></li>`).join("") || "<li>Ninguna.</li>"}</ul>
      </div>
    </div>`;
}
