interface Future {
  role: string;
  role_explanation: string;
  name: string;
  region: string;
  affinity: number;
  resilience: number;
  distance_from_anchor: number;
  dominant_facet: string;
  stability?: number;
  counterfactual: { facet: string; priority_shift: number | null };
  evidence?: { Review: string; Type: string; Polarity: number };
}

const escapeHtml = (value: unknown) => String(value)
  .replaceAll("&", "&amp;")
  .replaceAll("<", "&lt;")
  .replaceAll(">", "&gt;")
  .replaceAll('"', "&quot;")
  .replaceAll("'", "&#039;");

export default function cardMarkup(future: Future, index: number): string {
  const shift = future.counterfactual.priority_shift;
  let explanation = `Su fuerza principal está en ${future.dominant_facet}.`;
  if (future.role === "Ancla") {
    explanation = `Es la coincidencia directa. Su eje dominante es ${future.dominant_facet}.`;
  } else if (shift === null) {
    explanation = `Abre una ruta diferente a través de ${future.counterfactual.facet}.`;
  } else if (shift === 0) {
    explanation = `Supera al Ancla si miramos ${future.counterfactual.facet} por separado.`;
  } else {
    explanation = `Mover ${shift}% de atención hacia ${future.counterfactual.facet} abre este futuro.`;
  }

  const review = future.evidence?.Review ?? "";
  const evidence = future.evidence ? `
    <details>
      <summary>Ver evidencia real</summary>
      <p>“${escapeHtml(review.slice(0, 210))}${review.length > 210 ? "…" : ""}”</p>
      <small>${escapeHtml(future.evidence.Type)} · valoración ${future.evidence.Polarity}/5</small>
    </details>` : "";

  return `
    <article class="future-card ${index === 0 ? "anchor" : ""}" data-role="${escapeHtml(future.role)}">
      <div class="card-topline">
        <span class="role-pill"><span class="role-dot"></span>${escapeHtml(future.role)}</span>
        <span class="future-number">0${index + 1}</span>
      </div>
      <div class="card-heading">
        <div><p class="region">${escapeHtml(future.region)}</p><h3>${escapeHtml(future.name)}</h3></div>
        <div class="affinity-orb" style="--score:${future.affinity * 3.6}deg"><span>${Math.round(future.affinity)}</span></div>
      </div>
      <p class="role-description">${escapeHtml(future.role_explanation)}</p>
      <div class="signal-grid">
        <div><span>Resiliencia</span><strong>${future.resilience.toFixed(1)}</strong></div>
        <div><span>Distancia</span><strong>${future.distance_from_anchor.toFixed(1)}</strong></div>
        <div><span>Eje dominante</span><strong>${escapeHtml(future.dominant_facet)}</strong></div>
        <div><span>Estabilidad</span><strong>${future.stability === undefined ? "—" : `${future.stability.toFixed(1)}%`}</strong></div>
      </div>
      <div class="counterfactual"><span class="branch-icon">↗</span><p>${escapeHtml(explanation)}</p></div>
      ${evidence}
    </article>`;
}
