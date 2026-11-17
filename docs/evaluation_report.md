# Reporte de evaluación automática

Generado: 2026-11-17T06:03:39.765336+00:00 · modo **léxico** · duración 67.9 s

> “5 mejores” = mayor afinidad temática con la consulta. “5 peores” = menor afinidad; no implica peor calidad. La calidad (polaridad con prior bayesiano) se reporta aparte.

## Rúbrica

| Criterio | Valor | Umbral | Estado |
|---|---|---|---|
| Precision@5 (gold, 11 consultas con ≥5 relevantes) | 0.927 | 0.7 | ✅ |
| Hit@1 (gold) | 1.000 | 0.8 | ✅ |
| Contrato 5 + 5 | 1.000 | 1.0 | ✅ |
| Determinismo | 1.000 | 1.0 | ✅ |
| Latencia p95 en proceso (s) | 0.294 | 3.0 | ✅ |

## Gold seed (heurístico, pendiente de adjudicación humana)

| Métrica | Sistema | Popularidad | Azar |
|---|---|---|---|
| precision_at_5 | 0.525 | 0.135 | 0.095 |
| capped_recall_at_5 | 0.882 | 0.182 | 0.143 |
| hit_at_1 | 1.000 | 0.175 | 0.096 |
| hit_at_5 | 1.000 | 0.375 | 0.371 |
| mrr | 1.000 | 0.278 | 0.238 |
| ndcg_at_5 | 0.908 | 0.162 | 0.119 |
| bottom_error_rate | 0.000 | 0.300 | 0.364 |
| exclusion_violation_rate | 0.000 | 1.000 | 0.422 |

### Ablación de señales

| Configuración | Hit@1 | MRR | nDCG@5 |
|---|---|---|---|
| solo_perfil_lexico | 1.000 | 1.000 | 0.882 |
| solo_evidencia | 0.800 | 0.882 | 0.753 |
| solo_cobertura | 1.000 | 1.000 | 0.901 |
| lexico_anterior | 1.000 | 1.000 | 0.889 |
| lexico_cobertura | 1.000 | 1.000 | 0.908 |
| bm25 | 0.950 | 0.963 | 0.810 |

## Holdout autosupervisado (sin consultas escritas a mano)

Cada reseña reservada (no vista al construir el índice) actúa como consulta con nombres de pueblo y región enmascarados; el destino correcto es su pueblo.

600 reseñas reservadas · modo léxico

| Métrica | Sistema | Popularidad | Azar |
|---|---|---|---|
| Hit@1 | 0.475 | 0.025 | 0.025 |
| Hit@5 | 0.708 | 0.125 | 0.125 |
| MRR | 0.587 | 0.107 | — |
| Pueblo correcto en bottom 5 | 0.015 | 0.125 | 0.125 |

Spearman(Hit@5 por pueblo, nº de reseñas) = 0.2487

## Sesgo por volumen de reseñas

Spearman(apariciones en top 5, nº de reseñas) = 0.1685 (p = 0.2985). Pueblos que nunca aparecen en el top 5 del gold: ninguno.

## Robustez y casos adversariales

Jaccard medio del top 5 ante variantes (sin acentos, relleno, orden): 0.9583 (mínimo 0.6667).

| Caso | Esperado | Resultado | OK |
|---|---|---|---|
| vacía | reject | rechazada: Describe al menos dos palabras sobre el viaje que buscas. | ✅ |
| una palabra | reject | rechazada: Describe al menos dos palabras sobre el viaje que buscas. | ✅ |
| solo stopwords | reject_or_low | rechazada: Ninguna palabra de la consulta aparece en las reseñas del train; el r | ✅ |
| fuera de vocabulario | reject_or_low | rechazada: Ninguna palabra de la consulta aparece en las reseñas del train; el r | ✅ |
| solo números | reject_or_low | rechazada: Ninguna palabra de la consulta aparece en las reseñas del train; el r | ✅ |
| emoji | serve | servida (confianza alta) | ✅ |
| html | serve | servida (confianza alta) | ✅ |
| muy larga | serve | servida (confianza alta) | ✅ |
| todo negado | reject | rechazada: La consulta solo contiene exclusiones; di también qué buscas. | ✅ |
| pueblo excluido | serve | servida (confianza alta) | ✅ |

## Consultas con fallos (Hit@5 = 0 o relevante en bottom)

- Ninguna.

## Evaluación humana ciega

Sin sesiones todavía: usa la pestaña Evaluación de la web.
