import { api } from "./api";
// Vista principal: 5 destinos más afines y 5 menos afines, con verificación contra el corpus.

export const escapeHtml = (value: unknown) => String(value ?? "")
  .replaceAll("&", "&amp;")
  .replaceAll("<", "&lt;")
  .replaceAll(">", "&gt;")
  .replaceAll('"', "&quot;")
  .replaceAll("'", "&#039;");

interface CorpusRow { term: string; rate: number; median_rate: number }
interface FacetRow { facet: string; rate?: number; median_rate?: number; reason: string }
interface Violation { constraint: string; rate: number; median_rate: number }
export interface Destination {
  rank: number;
  global_rank: number;
  town: string;
  name: string;
  region: string;
  score: number;
  matched_terms: string[];
  keywords: string[];
  corpus_check: CorpusRow[];
  review_count: number;
  quality_bayes: number;
  least_related: boolean;
  evidence?: { Review: string; Type: string; Polarity: number } | null;
  violation_evidence?: { Review: string; Type: string; Polarity: number } | null;
  satisfied: FacetRow[];
  weak: FacetRow[];
  unsupported: FacetRow[];
  violated: Violation[];
  explanation: string;
}
export interface RecommendResult {
  query: string;
  mode: string;
  token: string;
  excluded_concepts: string;
  confidence: { level: string; note: string; unknown_terms: string[] };
  score_note: string;
  contradictions: string[];
  constraints: { constraint: string; towns_with_evidence: number }[];
  top: Destination[];
  bottom: Destination[];
}

const typeLabel: Record<string, string> = {
  Attractive: "Atracción", Hotel: "Hotel", Restaurant: "Restaurante",
};

function corpusChips(rows: CorpusRow[]): string {
  if (!rows.length) return "";
  return `<ul class="corpus-chips" aria-label="Menciones en reseñas del train">${rows.map((row) => {
    const lift = row.median_rate > 0 ? row.rate / row.median_rate : row.rate > 0 ? Infinity : 0;
    const strong = lift >= 2 && row.rate >= 0.5;
    return `<li class="${strong ? "strong" : ""}"><b>${escapeHtml(row.term)}</b>
      ${row.rate.toFixed(row.rate < 1 ? 2 : 1)}% <span>· mediana ${row.median_rate.toFixed(2)}%</span></li>`;
  }).join("")}</ul>`;
}

function facetList(title: string, rows: FacetRow[], css: string): string {
  if (!rows.length) return "";
  return `<p class="facets ${css}"><b>${title}:</b> ${rows.map((r) =>
    `${escapeHtml(r.facet)}${r.rate !== undefined ? ` <span>(${r.rate}% · mediana ${r.median_rate}%)</span>` : ""}`).join(", ")}</p>`;
}

export function destinationCard(item: Destination): string {
  const review = item.evidence?.Review ?? "";
  const why = item.least_related
    ? `<p class="why">Sus reseñas hablan sobre todo de: ${escapeHtml(item.keywords.slice(0, 5).join(", "))}.</p>`
    : item.matched_terms.length
      ? `<p class="why">Coincide en: ${escapeHtml(item.matched_terms.join(", "))}.</p>`
      : "";
  const violation = item.violation_evidence ? `
    <details open>
      <summary>Reseña que muestra la restricción incumplida</summary>
      <blockquote>“${escapeHtml(item.violation_evidence.Review.slice(0, 260))}${item.violation_evidence.Review.length > 260 ? "…" : ""}”</blockquote>
    </details>` : "";
  const evidence = item.evidence ? `
    <details>
      <summary>Reseña de respaldo</summary>
      <blockquote>“${escapeHtml(review.slice(0, 260))}${review.length > 260 ? "…" : ""}”</blockquote>
      <small>${escapeHtml(typeLabel[item.evidence.Type] ?? item.evidence.Type)} · polaridad ${item.evidence.Polarity}/5</small>
    </details>` : "";
  return `
    <li class="dest-card ${item.least_related ? "least" : ""}">
      <div class="dest-head">
        <span class="dest-rank" title="Posición global entre 40">${item.global_rank}</span>
        <div><h4>${escapeHtml(item.name)}</h4><p>${escapeHtml(item.region)}</p></div>
        <span class="affinity" title="Afinidad relativa entre los 40 destinos">
          <span class="visually-hidden">Afinidad</span>${item.score.toFixed(0)}
        </span>
      </div>
      <div class="affinity-bar" role="presentation"><i style="width:${Math.max(2, item.score)}%"></i></div>
      <p class="explanation">${escapeHtml(item.explanation)}</p>
      ${item.violated.length ? `<p class="facets violated"><b>Restricciones incumplidas:</b> ${item.violated.map((v) =>
        `${escapeHtml(v.constraint)} <span>(${v.rate}% de sus reseñas · mediana ${v.median_rate}%)</span>`).join(", ")}</p>` : ""}
      ${facetList("Satisface", item.satisfied, "ok")}
      ${facetList("Respaldo débil", item.weak, "weak")}
      ${facetList("Sin evidencia", item.unsupported, "none")}
      ${why}
      <p class="quality">Calidad percibida <b>${item.quality_bayes.toFixed(2)}/5</b> ·
        ${item.review_count.toLocaleString("es-MX")} reseñas <span>(no influye en el orden)</span></p>
      ${violation}${evidence}
    </li>`;
}

export function renderClassic(data: RecommendResult): string {
  const confidence = `<p class="confidence ${escapeHtml(data.confidence.level)}">
    Confianza <b>${escapeHtml(data.confidence.level)}</b> · ${escapeHtml(data.confidence.note)}
    ${data.confidence.unknown_terms.length ? ` Sin coincidencia en el corpus: ${escapeHtml(data.confidence.unknown_terms.join(", "))}.` : ""}
    <span>Motor: ${escapeHtml(data.mode)}</span></p>
    <p class="score-note">${escapeHtml(data.score_note)}${data.constraints.map((c) =>
      ` Restricción “${escapeHtml(c.constraint)}”: ${c.towns_with_evidence} pueblos con evidencia explícita de incumplirla.`).join("")}</p>`;
  const excluded = data.excluded_concepts
    ? `<p class="excluded">Exclusión aplicada: <b>${escapeHtml(data.excluded_concepts)}</b></p>` : "";
  return `${confidence}${excluded}
    <div class="classic-columns">
      <section aria-labelledby="top-title">
        <h3 id="top-title">TOP 5 — MAYOR COMPATIBILIDAD</h3>
        <p class="bottom-note">Los 5 mejores para esta consulta (posiciones 1–5 de 40).</p>
        <ol class="dest-list">${data.top.map(destinationCard).join("")}</ol>
      </section>
      <section aria-labelledby="bottom-title">
        <h3 id="bottom-title">BOTTOM 5 — MENOR COMPATIBILIDAD</h3>
        <p class="bottom-note">Los 5 peores para esta consulta, del menos compatible (40) hacia arriba. No significa peor destino ni peor servicio.</p>
        <ol class="dest-list">${data.bottom.map(destinationCard).join("")}</ol>
      </section>
    </div>`;
}

export async function postJson<T>(url: string, body: unknown): Promise<T> {
  const response = await fetch(api(url), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `Error ${response.status}`);
  return data as T;
}
