# Experimentos de recuperación (lexical)

Holdout: 320 consultas de calibración, 280 de prueba. Rejilla: 286 combinaciones (paso 0.1). Duración 47.8 s.

| Sistema | Pesos | MRR (prueba) | Hit@1 | Hit@5 | nDCG@5 | En bottom 5 |
|---|---|---|---|---|---|---|
| tfidf_perfil | lexical=1.00 | 0.495 | 0.368 | 0.621 | 0.505 | 0.021 |
| evidencia_reseñas | evidence=1.00 | 0.316 | 0.196 | 0.418 | 0.308 | 0.029 |
| bm25_medio | bm25=1.00 | 0.466 | 0.336 | 0.604 | 0.477 | 0.025 |
| lexico_anterior (0.68·tfidf + 0.32·evid) | lexical=0.68, evidence=0.32 | 0.511 | 0.396 | 0.614 | 0.513 | 0.011 |
| bm25 + evidencia (0.68/0.32) | evidence=0.32, bm25=0.68 | 0.500 | 0.386 | 0.614 | 0.506 | 0.018 |
| cobertura_facetas | coverage=1.00 | 0.541 | 0.432 | 0.671 | 0.553 | 0.021 |
| lexico_cobertura (0.30/0.20/0.50 cob) | lexical=0.30, evidence=0.20, coverage=0.50 | 0.575 | 0.471 | 0.682 | 0.582 | 0.018 |
| ajustado_en_calibracion | lexical=0.20, evidence=0.10, bm25=0.10, coverage=0.60 | 0.568 | 0.457 | 0.675 | 0.574 | 0.021 |

Ajustado vs lexico_anterior (0.68·tfidf + 0.32·evid) en prueba: ΔMRR = +0.0563 (IC95 +0.0241 a +0.0891); ΔHit@5 = +0.0607 (IC95 +0.0214 a +0.1000).

Léxico en producción (con cobertura de facetas) vs léxico anterior en prueba: ΔMRR = +0.0634 (IC95 +0.0353 a +0.0932); ΔHIT_AT_1 = +0.0750 (IC95 +0.0286 a +0.1214).
