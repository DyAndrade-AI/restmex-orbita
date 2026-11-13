interface Destination {
  name: string;
  region: string;
  share: number;
  share_se: number;
  share_ci95: [number, number];
  count: number;
  identity: string;
  dominant_facet: string;
  affinity: number;
  sacrifice: { facet: string; priority_shift: number | null };
  evidence?: { Review: string; Type: string; Polarity: number };
}

interface Branch {
  keep: string;
  sacrifice: string;
  survivors: number;
  eliminated: number;
  destination: string;
}

const escapeHtml = (value: unknown) => String(value)
  .replaceAll("&", "&amp;")
  .replaceAll("<", "&lt;")
  .replaceAll(">", "&gt;")
  .replaceAll('"', "&quot;")
  .replaceAll("'", "&#039;");

export function basinMarkup(destination: Destination, index: number): string {
  const shift = destination.sacrifice.priority_shift;
  const sacrifice = index === 0
    ? "Es el futuro con mayor gravedad dentro del multiverso actual."
    : shift === null
      ? `Su portal se abre desde ${destination.sacrifice.facet}.`
      : `Requiere mover ${shift}% de atención hacia ${destination.sacrifice.facet}.`;
  const review = destination.evidence?.Review ?? "";
  return `
    <article class="basin-card" style="--basin-index:${index}">
      <div class="basin-top"><span>0${index + 1}</span><b>${destination.share.toFixed(2)}%</b></div>
      <small class="basin-ci">IC95 ${destination.share_ci95[0].toFixed(2)}–${destination.share_ci95[1].toFixed(2)} % · EE ${destination.share_se.toFixed(2)} pp</small>
      <p>${escapeHtml(destination.region)}</p>
      <h4>${escapeHtml(destination.name)}</h4>
      <div class="basin-meter"><i style="width:${Math.max(destination.share, 2)}%"></i></div>
      <strong>${escapeHtml(destination.identity)}</strong>
      <small>${escapeHtml(sacrifice)}</small>
      ${destination.evidence ? `<details><summary>Evidencia del futuro</summary><p>“${escapeHtml(review.slice(0, 180))}${review.length > 180 ? "…" : ""}”</p></details>` : ""}
    </article>`;
}

export function branchMarkup(branch: Branch): string {
  return `
    <button type="button" data-keep="${escapeHtml(branch.keep)}" data-sacrifice="${escapeHtml(branch.sacrifice)}">
      <span>Sacrificar <b>${escapeHtml(branch.sacrifice)}</b></span>
      <strong>Conservar ${escapeHtml(branch.keep)} <i>→</i></strong>
      <small>${branch.survivors.toLocaleString("es-MX")} futuros sobreviven · gravitan hacia ${escapeHtml(branch.destination)}</small>
    </button>`;
}

export function decisionTrailMarkup(choices: Array<{ keep: string; sacrifice: string }>): string {
  if (!choices.length) return "<span>Aún no has sacrificado ninguna preferencia.</span>";
  return choices.map((choice, index) =>
    `<span><b>Decisión ${index + 1}</b> ${escapeHtml(choice.keep)} sobre <s>${escapeHtml(choice.sacrifice)}</s></span>`
  ).join("");
}

export function trajectoryMarkup(step: { step: number; name: string; region: string; awakens: string }): string {
  return `
    <div class="trajectory-step">
      <span>0${step.step}</span>
      <i></i>
      <div><small>${escapeHtml(step.region)}</small><strong>${escapeHtml(step.name)}</strong><p>Despierta: ${escapeHtml(step.awakens)}</p></div>
    </div>`;
}

export function bottomMultiverseMarkup(rows: any[]): string {
  return rows.map((row) => `
    <li class="${row.blocked_by_constraint ? "blocked" : ""}">
      <b>#${row.base_rank} ${escapeHtml(row.name)}</b>
      ${row.blocked_by_constraint
        ? `<span>Excluido por la restricción en el 100 % de los futuros.</span>`
        : `<span>Sigue en el bottom 5 en ${row.stays_bottom5_pct}% de los futuros · entra al top 5 en ${row.enters_top5_pct}% · gana en ${row.wins_pct}% · desviación de posición ${row.rank_sd}</span>
           ${row.competitive_when ? `<small>${escapeHtml(row.competitive_when)}</small>` : ""}`}
    </li>`).join("");
}

export function rankingShiftMarkup(before: any[], after: any[]): string {
  const prev = new Map(before.map((d: any, i: number) => [d.town, { i, share: d.share }]));
  return after.map((d: any, i: number) => {
    const old = prev.get(d.town);
    const move = old === undefined ? "nuevo" : old.i === i ? "=" : old.i > i ? `↑${old.i - i}` : `↓${i - old.i}`;
    const delta = old === undefined ? "" : ` (${(d.share - old.share >= 0 ? "+" : "")}${(d.share - old.share).toFixed(2)} pp)`;
    return `<li><b>${i + 1}. ${escapeHtml(d.name)}</b> <span>${d.share.toFixed(2)}%${delta}</span> <i>${move}</i></li>`;
  }).join("");
}
