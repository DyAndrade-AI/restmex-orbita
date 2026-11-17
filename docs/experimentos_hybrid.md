# Experimentos de recuperación (hybrid)

Holdout: 320 consultas de calibración, 280 de prueba. Rejilla: 1001 combinaciones (paso 0.1). Duración 43.8 s.

| Sistema | Pesos | MRR (prueba) | Hit@1 | Hit@5 | nDCG@5 | En bottom 5 |
|---|---|---|---|---|---|---|
| tfidf_perfil | lexical=1.00 | 0.495 | 0.368 | 0.621 | 0.505 | 0.021 |
| evidencia_reseñas | evidence=1.00 | 0.316 | 0.196 | 0.418 | 0.308 | 0.029 |
| bm25_medio | bm25=1.00 | 0.466 | 0.336 | 0.604 | 0.477 | 0.025 |
| lexico_anterior (0.68·tfidf + 0.32·evid) | lexical=0.68, evidence=0.32 | 0.511 | 0.396 | 0.614 | 0.513 | 0.011 |
| bm25 + evidencia (0.68/0.32) | evidence=0.32, bm25=0.68 | 0.500 | 0.386 | 0.614 | 0.506 | 0.018 |
| cobertura_facetas | coverage=1.00 | 0.541 | 0.432 | 0.671 | 0.553 | 0.021 |
| lexico_cobertura (0.30/0.20/0.50 cob) | lexical=0.30, evidence=0.20, coverage=0.50 | 0.575 | 0.471 | 0.682 | 0.582 | 0.018 |
| e5 | semantic=1.00 | 0.253 | 0.146 | 0.314 | 0.233 | 0.046 |
| hibrido_original (0.38/0.17/0.45) | lexical=0.38, evidence=0.17, semantic=0.45 | 0.468 | 0.346 | 0.600 | 0.478 | 0.011 |
| ajustado_en_calibracion | evidence=0.20, bm25=0.30, semantic=0.10, coverage=0.40 | 0.572 | 0.461 | 0.675 | 0.578 | 0.018 |

Ajustado vs hibrido_original (0.38/0.17/0.45) en prueba: ΔMRR = +0.1044 (IC95 +0.0714 a +0.1386); ΔHit@5 = +0.0750 (IC95 +0.0321 a +0.1214).

Léxico en producción (con cobertura de facetas) vs léxico anterior en prueba: ΔMRR = +0.0634 (IC95 +0.0353 a +0.0932); ΔHIT_AT_1 = +0.0750 (IC95 +0.0286 a +0.1214).
