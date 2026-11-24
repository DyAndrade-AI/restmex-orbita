# Validación sintética de Strange Loop y Órbita

> Generado por `scripts/validate_orbit.py` desde `artifacts/orbit_validation.json`. No editar a mano.
>
> **Tipo de evidencia: usuarios simulados.** No es evaluación humana: no hay juicios humanos en este experimento.

- Fecha: 2026-11-24 00:35 · modo: léxico (sin E5, igual que audit_multiverse) · usuarios por consulta: 150 · semilla del usuario i = índice de la consulta · duración 155.6 s
- Consultas: 64 (banco + auditoría + ejemplo del video); las de una sola faceta no tienen preguntas y se omiten.

| Población | Consultas | Ancla Hit@1 | SL 0 preg. | SL 1 preg. info / azar | SL 3 preg. info / azar | Hit@5 (3 preg.) | Mejora 3 vs 0 (IC95) | Info − azar, 1 preg. (IC95) | Azar > info (1 preg.) | Preguntar empeora (3) |
|---|---|---|---|---|---|---|---|---|---|---|
| prior | 48 | 69.0 % | 84.5 % | 91.3 % / 89.8 % | 92.4 % / 92.2 % | 99.9 % | +7.9 pp [4.2, 12.2] | +1.5 pp [0.3, 2.9] | 1 | 0 |
| concentrada | 48 | 54.0 % | 62.3 % | 70.2 % / 68.1 % | 71.4 % / 71.2 % | 98.1 % | +9.1 pp [4.8, 14.0] | +2.1 pp [0.9, 3.6] | 0 | 0 |
| relevancia_0.50 | 48 | 81.4 % | 82.8 % | 86.4 % / 85.5 % | 86.8 % / 86.8 % | 99.3 % | +3.9 pp [-1.7, 9.8] | +0.9 pp [0.1, 2.0] | 1 | 3 |
| rendimientos_decrecientes | 48 | 70.8 % | 83.1 % | 91.6 % / 89.6 % | 91.5 % / 91.3 % | 100.0 % | +8.4 pp [4.2, 13.3] | +1.9 pp [0.6, 3.6] | 1 | 1 |
| ruido_15 | 48 | 69.0 % | 84.5 % | 88.4 % / 87.4 % | 88.8 % / 88.8 % | 99.6 % | +4.3 pp [1.9, 7.1] | +1.0 pp [-0.1, 2.2] | 2 | 2 |
| ruido_30 | 48 | 69.0 % | 84.5 % | 86.0 % / 85.2 % | 85.7 % / 85.4 % | 99.4 % | +1.2 pp [-0.5, 3.0] | +0.7 pp [-0.1, 1.7] | 2 | 5 |

Poblaciones: ver `POPULATIONS` en `src/restmex_recommender/orbit_eval.py`. *prior* usa la misma utilidad y el mismo Dirichlet que el modelo (caso optimista, coherencia interna); las demás rompen un supuesto a propósito. Hit@1 = el destino más frecuente del multiverso restante es el ideal oculto del usuario.

## Sensibilidad de los pesos de rol (±50 %, 200 sorteos por consulta, riesgo 0.55)

| Rol | Conserva el mismo destino |
|---|---|
| Ancla | 100.0 % |
| Superviviente | 93.1 % |
| Portal | 83.9 % |
| Especialista | 84.6 % |
| Comodín | 75.3 % |

Jaccard medio de la cartera: 0.9041 · cartera idéntica: 72.3 % de los sorteos.
