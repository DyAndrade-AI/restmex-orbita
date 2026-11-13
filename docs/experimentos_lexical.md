# Experimentos de recuperación (lexical)

Holdout: 320 consultas de calibración, 280 de prueba. Rejilla: 66 combinaciones (paso 0.1). Duración 44.8 s.

| Sistema | Pesos | MRR (prueba) | Hit@1 | Hit@5 | nDCG@5 | En bottom 5 |
|---|---|---|---|---|---|---|
| tfidf_perfil | lexical=1.00 | 0.495 | 0.368 | 0.621 | 0.505 | 0.021 |
| evidencia_reseñas | evidence=1.00 | 0.316 | 0.196 | 0.418 | 0.308 | 0.029 |
| bm25_medio | bm25=1.00 | 0.466 | 0.336 | 0.604 | 0.477 | 0.025 |
| lexico_actual (0.68·tfidf + 0.32·evid) | lexical=0.68, evidence=0.32 | 0.511 | 0.396 | 0.614 | 0.513 | 0.011 |
| bm25 + evidencia (0.68/0.32) | evidence=0.32, bm25=0.68 | 0.500 | 0.386 | 0.614 | 0.506 | 0.018 |
| ajustado_en_calibracion | lexical=0.50, evidence=0.20, bm25=0.30 | 0.522 | 0.400 | 0.661 | 0.538 | 0.018 |

Ajustado vs lexico_actual (0.68·tfidf + 0.32·evid) en prueba: ΔMRR = +0.0105 (IC95 -0.0095 a +0.0314); ΔHit@5 = +0.0464 (IC95 +0.0214 a +0.0750).
